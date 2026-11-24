"""使用 Reloc3r 和 Aachen 预计算检索候选执行绝对位姿估计。

每个 query 直接读取 pairs 文件中排名靠前的 db 图像，不重新计算检索分数。
Reloc3r 估计 query 到 db 的相对位姿；利用 COLMAP SFM 中 db 图像的世界位姿，
经过旋转一致性筛选、射线一致性筛选和运动平均得到 query 在 SFM 坐标系中的位姿。
"""

import argparse
from pathlib import Path

import numpy as np
import pycolmap
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from reloc import rotation_consensus, translation_ray_consensus
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.device import to_numpy
from reloc3r.utils.image import load_images


def read_pairs(path):
    """读取 query/db pairs，并保持文件中候选数据库图的原始排序。"""
    pairs = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError('Invalid pairs line {}: {}'.format(line_no, line.strip()))
            query_name, db_name = fields[:2]
            pairs.setdefault(query_name, []).append(db_name)
    return pairs


def load_sfm_poses(model_dir):
    """从 COLMAP 二进制重建读取每张已注册 db 图像的 world-to-camera 位姿。"""
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    poses_w2c = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        # PyCOLMAP 的 cam_from_world() 即 COLMAP world-to-camera 变换，返回 3x4。
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        if np.isfinite(matrix).all():
            poses_w2c[image.name.replace('\\', '/')] = matrix
    return poses_w2c


def main():
    parser = argparse.ArgumentParser(description='Reloc3r localization on Aachen Day-Night.')
    parser.add_argument('--dataset_dir', type=str,
                        default=r'D:\Datasets\Aachen-Day-Night\dataset\images_upright',
                        help='Aachen images_upright 根目录')
    parser.add_argument('--pairs', type=str,
                        default=r'D:\project\Hierarchical-Localization\outputs\aachen\pairs-query-megaloc10.txt',
                        help='按检索得分排序的 query/db pairs 文件')
    parser.add_argument('--model_dir', type=str,
                        default=r'D:\project\Hierarchical-Localization\outputs\aachen\sfm_superpoint+lightglue',
                        help='COLMAP/PyCOLMAP SFM 二进制模型目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录；默认 <repo>/output/Aachen_full')
    parser.add_argument('--img_reso', type=str, default='512', choices=['224', '512'])
    parser.add_argument('--image_resize', choices=['stretch', 'pad'], default='pad',
                        help='仅处理竖屏图像：拉伸为正方形或左右补边；横屏图像不变')
    parser.add_argument('--topk', type=int, default=5,
                        help='使用 pairs 文件中每个 query 的前 K 个 db 候选')
    parser.add_argument('--rotation_threshold_deg', type=float, default=15.0,
                        help='旋转 medoid 筛选阈值（度），与 reloc.py 一致')
    parser.add_argument('--rotation_consensus', dest='rotation_consensus',
                        action='store_true', default=True,
                        help='开启旋转一致性筛选（默认开启）')
    parser.add_argument('--no_rotation_consensus', dest='rotation_consensus',
                        action='store_false', help='关闭旋转一致性筛选')
    parser.add_argument('--translation_parallel_threshold_deg', type=float, default=5.0,
                        help='夹角小于该阈值的射线对跳过距离计算（度）')
    parser.add_argument('--translation_consensus', dest='translation_consensus',
                        action='store_true', default=True,
                        help='开启平移射线一致性筛选（默认开启）')
    parser.add_argument('--no_translation_consensus', dest='translation_consensus',
                        action='store_false', help='关闭平移射线一致性筛选')
    parser.add_argument('--query_image', type=str, nargs='+', default=None,
                        help='只定位指定 query；可用 pairs 中的完整路径或文件名')
    # 一般要开启--pose_details
    parser.add_argument('--pose_details', action='store_true',
                        help='在 output_dir/pose_details 保存逐 query 诊断日志')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='是否使用自动混合精度')
    args = parser.parse_args()

    if args.topk < 1:
        parser.error('--topk must be positive')
    if not np.isfinite(args.rotation_threshold_deg) or args.rotation_threshold_deg < 0:
        parser.error('--rotation_threshold_deg must be finite and non-negative')
    if (not np.isfinite(args.translation_parallel_threshold_deg)
            or not 0.0 <= args.translation_parallel_threshold_deg <= 90.0):
        parser.error('--translation_parallel_threshold_deg must be between 0 and 90')

    script_dir = Path(__file__).resolve().parent
    if args.output_dir is None:
        args.output_dir = str(script_dir / 'output' / 'Aachen')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    # pairs 文件给定候选排名；这里仅按 query 分组和截取 top-K，不计算描述子相似度。
    pairs = read_pairs(args.pairs)
    if args.query_image:
        requested = set(args.query_image)
        pairs = {q: dbs for q, dbs in pairs.items()
                 if q in requested or Path(q).name in requested}
    if not pairs:
        raise ValueError('No query pairs selected from {}'.format(args.pairs))

    print('Loading Aachen SFM model...')
    map_poses_w2c = load_sfm_poses(args.model_dir)
    print('  registered map images: {}'.format(len(map_poses_w2c)))
    if not map_poses_w2c:
        raise ValueError('No registered image poses found in {}'.format(args.model_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Loading Reloc3r-{} model...'.format(args.img_reso))
    model = setup_reloc3r_relpose_model(model_args=args.img_reso, device=device)
    visloc = Reloc3rVisloc()

    # 查询图和数据库图可能跨 query 共用，缓存预处理后的 CPU 张量。
    image_cache = {}

    def get_image(image_name):
        """读取 Aachen 图像；只将竖屏图变成正方形，横屏和方形图不改动。"""
        image_path = dataset_dir / Path(image_name.replace('\\', '/'))
        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))
        key = str(image_path)
        if key not in image_cache:
            item = load_images([key], size=int(args.img_reso), verbose=False)[0]
            _, _, height, width = item['img'].shape
            if width < height:
                side = height
                if args.image_resize == 'stretch':
                    # stretch 会改变竖屏图像的比例，直接缩放到 side x side。
                    item['img'] = torch.nn.functional.interpolate(
                        item['img'], size=(side, side), mode='bilinear',
                        align_corners=False)
                else:
                    # pad 保持比例，仅在左右两侧补边，不改动图像有效区域。
                    pad_left = (side - width) // 2
                    pad_right = side - width - pad_left
                    item['img'] = torch.nn.functional.pad(
                        item['img'], (pad_left, pad_right, 0, 0), value=-1.0)
                item['true_shape'] = np.asarray([[side, side]], dtype=np.int32)
            image_cache[key] = item
        item = image_cache[key]
        return {'img': item['img'].to(device),
                'true_shape': torch.as_tensor(item['true_shape']).to(device)}

    def save_pose_details(query, pair_details, rotation_rows, medoid_name,
                          failure_reason=None, query_c2w=None, query_w2c=None):
        """保存相对位姿、旋转一致性筛选及最终定位结果。"""
        if not args.pose_details:
            return
        details_dir = output_dir / 'pose_details'
        details_dir.mkdir(parents=True, exist_ok=True)
        safe_name = query.replace('\\', '_').replace('/', '_')
        with (details_dir / (safe_name + '.txt')).open('w', encoding='utf-8') as f:
            f.write('Query image: {}\n'.format(query))
            f.write('Top-K: {}\n'.format(args.topk))
            f.write('Rotation consensus enabled: {}\n'.format(args.rotation_consensus))
            f.write('Rotation threshold: {:.6f} deg\n'.format(args.rotation_threshold_deg))
            f.write('Rotation medoid: {}\n'.format(medoid_name or 'N/A'))
            f.write('Rotation consensus (candidate / medoid error / retained):\n')
            for name, error, keep in rotation_rows:
                error_text = 'N/A' if error is None else '{:.6f} deg'.format(error)
                f.write('  {} / {} / {}\n'.format(name, error_text,
                                                   'yes' if keep else 'no'))
            f.write('Relative poses (query-to-database; normalized translation):\n')
            for row in pair_details:
                f.write('\nDatabase image: {}\n'.format(row['name']))
                f.write('  valid: {}\n'.format(row['valid']))
                if row['pose'] is not None:
                    f.write('{}\n'.format(np.array2string(row['pose'], precision=9)))
                if row['absolute_rotation'] is not None:
                    f.write('Estimated query rotation (world):\n')
                    f.write('{}\n'.format(np.array2string(
                        row['absolute_rotation'], precision=9)))
            if failure_reason:
                f.write('\nLocalization failure: {}\n'.format(failure_reason))
            if query_c2w is not None:
                f.write('\nFinal camera-to-world pose:\n')
                f.write('{}\n'.format(np.array2string(query_c2w, precision=9)))
            if query_w2c is not None:
                f.write('\nFinal world-to-camera pose:\n')
                f.write('{}\n'.format(np.array2string(query_w2c, precision=9)))

    def save_translation_details(query, names, keep_mask, details):
        """保存平移射线筛选的角度、最短距离、鲁棒分数和保留状态。"""
        if not args.pose_details:
            return
        details_dir = output_dir / 'pose_details'
        details_dir.mkdir(parents=True, exist_ok=True)
        safe_name = query.replace('\\', '_').replace('/', '_')
        with (details_dir / (safe_name + '.trans.txt')).open('w', encoding='utf-8') as f:
            f.write('Query image: {}\n'.format(query))
            f.write('Candidate count before translation filtering: {}\n'.format(len(names)))
            f.write('Translation consensus enabled: {}\n'.format(args.translation_consensus))
            f.write('Parallel-pair threshold: {:.6f} deg\n'.format(
                args.translation_parallel_threshold_deg))
            f.write('Pairwise acute ray angles (deg; NaN means unavailable):\n')
            f.write('{}\n'.format(np.array2string(details['angles'], precision=6)))
            f.write('Skipped near-parallel pairs:\n')
            for i, j, angle in details['skipped_pairs']:
                f.write('  {} <-> {}: {:.6f} deg\n'.format(names[i], names[j], angle))
            f.write('Pairwise half-ray distance matrix (NaN means invalid/skipped):\n')
            f.write('{}\n'.format(np.array2string(details['distances'], precision=6)))
            f.write('Minimum valid comparisons per candidate: {}\n'.format(
                details['min_comparisons']))
            f.write('Candidate scores (valid comparisons / median distance / status):\n')
            for i, name in enumerate(names):
                score = details['scores'][i]
                score_text = ('Insufficient Geometry' if not np.isfinite(score)
                              else '{:.6f}'.format(score))
                status = 'retained' if keep_mask[i] else 'rejected'
                f.write('  {} / {} / {} / {}\n'.format(
                    name, int(details['valid_counts'][i]), score_text, status))
            f.write('Median: {}\nMAD: {}\nRobust sigma: {}\nThreshold: {}\n'.format(
                details['median'], details['mad'], details['sigma'], details['threshold']))
            f.write('Retained candidates: {}\n'.format(
                ', '.join(n for n, keep in zip(names, keep_mask) if keep)))
            f.write('Rejected candidates: {}\n'.format(
                ', '.join(n for n, keep in zip(names, keep_mask) if not keep) or 'None'))

    results = []
    for query, db_names in tqdm(pairs.items(), desc='Localizing Aachen'):
        try:
            query_image = get_image(query)
        except FileNotFoundError as exc:
            print('Skip query {}: {}'.format(query, exc))
            continue

        poses_db_c2w = []
        poses_q2d = []
        candidate_rotations = []
        candidate_names = []
        pair_details = []

        # 每个数据库图提供一个绝对 SFM 位姿和 query->db 的相对位姿。
        for db_name in db_names[:args.topk]:
            normalized_name = db_name.replace('\\', '/')
            map_w2c = map_poses_w2c.get(normalized_name)
            if map_w2c is None:
                pair_details.append({'name': db_name, 'pose': None,
                                     'absolute_rotation': None, 'valid': False})
                continue
            try:
                db_image = get_image(db_name)
            except FileNotFoundError as exc:
                print('Skip map image {}: {}'.format(db_name, exc))
                pair_details.append({'name': db_name, 'pose': None,
                                     'absolute_rotation': None, 'valid': False})
                continue

            # view1=db, view2=query，因此 inference_relpose 返回 query 到 db 的变换。
            pose_q2d = to_numpy(inference_relpose(
                [db_image, query_image], model, device,
                use_amp=bool(args.amp))[0]).astype(np.float64)
            translation_norm = float(np.linalg.norm(pose_q2d[:3, 3]))
            if not np.isfinite(pose_q2d).all() or translation_norm < 1e-8:
                pair_details.append({'name': db_name, 'pose': pose_q2d.copy(),
                                     'absolute_rotation': None, 'valid': False})
                continue
            pose_q2d[:3, 3] /= translation_norm

            pose_db_c2w = np.linalg.inv(map_w2c)
            query_rotation = (pose_db_c2w @ pose_q2d)[:3, :3]
            pair_details.append({'name': db_name, 'pose': pose_q2d.copy(),
                                 'absolute_rotation': query_rotation.copy(), 'valid': True})
            poses_db_c2w.append(pose_db_c2w)
            poses_q2d.append(pose_q2d)
            candidate_rotations.append(query_rotation)
            candidate_names.append(db_name)

        if len(candidate_names) < 2:
            reason = 'fewer than two valid candidates'
            print('Skip {}: {}'.format(query, reason))
            save_pose_details(query, pair_details, [], None, reason)
            continue

        # 旋转 medoid 与原 reloc.py 相同；关闭开关时所有有效候选通过。
        distances, medoid_idx, rotation_keep = rotation_consensus(
            candidate_rotations, args.rotation_threshold_deg)
        medoid_name = candidate_names[medoid_idx]
        rotation_rows = [(name, float(distances[medoid_idx, i]), bool(rotation_keep[i]))
                         for i, name in enumerate(candidate_names)]
        if not args.rotation_consensus:
            rotation_keep = np.ones(len(candidate_names), dtype=bool)
            rotation_rows = [(name, None, True) for name in candidate_names]
        if np.count_nonzero(rotation_keep) < 2:
            reason = 'rotation consensus retained fewer than two candidates'
            print('Skip {}: {} (nearest medoid errors: {})'.format(
                query, reason,
                np.array2string(np.sort(distances[medoid_idx])[:5], precision=2)))
            save_pose_details(query, pair_details, rotation_rows, medoid_name, reason)
            continue
        poses_db_c2w = [p for p, keep in zip(poses_db_c2w, rotation_keep) if keep]
        poses_q2d = [p for p, keep in zip(poses_q2d, rotation_keep) if keep]
        candidate_names = [n for n, keep in zip(candidate_names, rotation_keep) if keep]

        # 由 map 相机中心和相对平移方向构造全局坐标中的半射线。
        centers = np.asarray([pose[:3, 3] for pose in poses_db_c2w], dtype=np.float64)
        directions = np.asarray([
            pose_db[:3, :3] @ pose_q[:3, 3]
            for pose_db, pose_q in zip(poses_db_c2w, poses_q2d)], dtype=np.float64)
        translation_keep, translation_details = translation_ray_consensus(
            centers, directions, candidate_names,
            parallel_threshold_deg=args.translation_parallel_threshold_deg)
        if not args.translation_consensus:
            translation_keep = np.ones(len(candidate_names), dtype=bool)
        save_translation_details(query, candidate_names, translation_keep, translation_details)
        poses_db_c2w = [p for p, keep in zip(poses_db_c2w, translation_keep) if keep]
        poses_q2d = [p for p, keep in zip(poses_q2d, translation_keep) if keep]
        if len(poses_db_c2w) < 2:
            reason = 'translation consensus retained fewer than two candidates'
            print('Skip {}: {}'.format(query, reason))
            save_pose_details(query, pair_details, rotation_rows, medoid_name, reason)
            continue

        try:
            pose_c2w = visloc.motion_averaging(poses_db_c2w, poses_q2d)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            print('Skip {}: motion averaging failed: {}'.format(query, exc))
            save_pose_details(query, pair_details, rotation_rows, medoid_name, str(exc))
            continue

        pose_w2c = np.linalg.inv(pose_c2w)
        save_pose_details(query, pair_details, rotation_rows, medoid_name,
                          query_c2w=pose_c2w, query_w2c=pose_w2c)
        results.append({'name': query, 'c2w': pose_c2w, 'w2c': pose_w2c,
                        'top_names': db_names[:args.topk],
                        'retained_count': len(poses_db_c2w)})

    if not results:
        raise RuntimeError('No Aachen queries could be localized.')

    # 输出格式兼容 COLMAP 的文本图像位姿：名字、四元数 qw qx qy qz、平移 tx ty tz。
    pose_path = output_dir / 'query_poses.txt'
    with pose_path.open('w', encoding='utf-8') as f:
        f.write('# Estimated query poses (world-to-camera, Aachen SFM coordinates)\n')
        f.write('# name QW QX QY QZ TX TY TZ retained_candidates\n')
        for result in results:
            pose = result['w2c']
            qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            tx, ty, tz = pose[:3, 3]
            f.write('{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {}\n'.format(
                result['name'], qw, qx, qy, qz, tx, ty, tz,
                result['retained_count']))
    print('Localized {} queries; poses saved to {}'.format(len(results), pose_path))
    print('Aachen query ground-truth poses were not loaded; no pose-error metrics were computed.')


if __name__ == '__main__':
    main()
