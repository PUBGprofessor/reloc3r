"""使用 Reloc3r 和预计算 NetVLAD 候选对完成 InLoc 查询图像定位。

InLoc 数据库图是从扫描点云生成的透视 cutout。脚本直接读取 pairs 文件中的
检索排序，不重新计算全局描述子相似度；地图姿态通过 cutout 自带的 XYZcut
和 DUC 扫描对齐矩阵恢复，之后执行相对位姿估计、一致性筛选和射线交会。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.io import loadmat
from tqdm import tqdm

from reloc import rotation_consensus, translation_ray_consensus
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.device import to_numpy
from reloc3r.utils.image import load_images


def read_pairs(path):
    """读取 pairs 文件，保留每个 query 下数据库图像原有的检索顺序。"""
    pairs = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError('Invalid pair line {}: {}'.format(line_no, line.strip()))
            query, database = fields[:2]
            pairs.setdefault(query, []).append(database)
    return pairs


def scan_to_world(dataset_dir, database_name):
    """读取该 cutout 所属 DUC 扫描坐标系到 InLoc 全局坐标系的变换。"""
    parts = database_name.replace('\\', '/').split('/')
    if len(parts) < 4 or parts[0] != 'cutouts':
        raise ValueError('Expected cutouts/DUC1|DUC2/<scan>/<image>, got {}'.format(database_name))
    floor, scan_id, image_name = parts[1], parts[2], parts[3]
    alignment = Path(dataset_dir) / '{}_alignment'.format(floor) / 'transformations'
    transform_path = alignment / '{}_trans_{}.txt'.format(image_name[:3], scan_id)
    if not transform_path.is_file():
        raise FileNotFoundError('Missing scan alignment: {}'.format(transform_path))
    # DUC transformation 文件前 7 行是说明文字，随后 4 行是 4x4 齐次矩阵。
    lines = transform_path.read_text(encoding='utf-8').splitlines()
    matrix = np.asarray([np.fromstring(lines[i], sep=' ') for i in range(7, 11)], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid 4x4 alignment in {}'.format(transform_path))
    return matrix, image_name


def cutout_camera_to_world(dataset_dir, database_name):
    """由 XYZcut 的像素-三维点对应关系恢复 cutout 相机到全局坐标系的位姿。

    不能仅凭文件名里的 yaw/pitch 直接拼欧拉角，因为它们不包含相机坐标轴
    与扫描坐标轴的完整约定。XYZcut 中每个有效像素都给出该像素所见点在扫描
    坐标系中的坐标；cutout 相机中心就是原始扫描全景中心（扫描坐标原点）。
    根据 60 度水平视场角构造像素射线，再用 Kabsch/SVD 求相机射线到扫描点
    方向的最佳旋转。最后左乘 DUC 扫描到全局坐标系的旋转和平移。
    """
    T_world_scan, image_name = scan_to_world(dataset_dir, database_name)
    image_path = Path(dataset_dir) / Path(database_name.replace('\\', '/'))
    mat_path = Path(str(image_path) + '.mat')
    if not mat_path.is_file():
        raise FileNotFoundError('Missing cutout XYZ data: {}'.format(mat_path))
    mat = loadmat(str(mat_path))
    if 'XYZcut' not in mat:
        raise ValueError('XYZcut array is missing in {}'.format(mat_path))
    xyz = np.asarray(mat['XYZcut'], dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError('Unexpected XYZcut shape {} in {}'.format(xyz.shape, mat_path))

    # 取稀疏网格以降低读取后拟合的计算量；XYZcut 是扫描坐标中的点坐标。
    height, width, _ = xyz.shape
    step = max(1, min(height, width) // 80)
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    points = xyz[0:height:step, 0:width:step].reshape(-1, 3)
    pixels_x = xs.reshape(-1)
    pixels_y = ys.reshape(-1)
    valid = np.isfinite(points).all(axis=1)
    points, pixels_x, pixels_y = points[valid], pixels_x[valid], pixels_y[valid]
    point_norms = np.linalg.norm(points, axis=1)
    valid = point_norms > 1e-6
    points, pixels_x, pixels_y, point_norms = (
        points[valid], pixels_x[valid], pixels_y[valid], point_norms[valid])
    if len(points) < 20:
        raise ValueError('Too few valid XYZcut samples in {}'.format(mat_path))

    # InLoc perspective cutout 的水平视场角为 60 度，采用方形像素模型。
    focal = width / (2.0 * np.tan(np.deg2rad(30.0)))
    rays_camera = np.column_stack((
        (pixels_x - (width - 1.0) / 2.0) / focal,
        (pixels_y - (height - 1.0) / 2.0) / focal,
        np.ones(len(points))))
    rays_camera /= np.linalg.norm(rays_camera, axis=1, keepdims=True)
    rays_scan = points / point_norms[:, None]

    # 求 R，使 R @ ray_camera ~= ray_scan。SVD 解正交 Procrustes 问题。
    covariance = rays_scan.T @ rays_camera
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    R_scan_camera = u @ correction @ vt

    # XYZcut 的来源是以扫描全景中心为投影中心，因此相机中心为扫描原点。
    T_world_camera = np.eye(4, dtype=np.float64)
    T_world_camera[:3, :3] = T_world_scan[:3, :3] @ R_scan_camera
    T_world_camera[:3, 3] = T_world_scan[:3, 3]
    return T_world_camera


def main():
    parser = argparse.ArgumentParser(description='Reloc3r localization on InLoc.')
    parser.add_argument('--dataset_dir', default=r'D:\Datasets\InLoc')
    parser.add_argument('--pairs', default=r'D:\project\Hierarchical-Localization\pairs\inloc\pairs-query-netvlad20-local.txt')
    parser.add_argument('--output_dir', default=r".\output\InLoc")
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--img_reso', choices=['224', '512'], default='512')
    parser.add_argument('--image_resize', choices=['stretch', 'pad'], default='pad')
    parser.add_argument('--rotation_threshold_deg', type=float, default=15.0,
                        help='medoid 旋转一致性角度阈值（度；与方法.md 一致）')
    parser.add_argument('--rotation_consensus', dest='rotation_consensus',
                        action='store_true', default=True,
                        help='开启旋转一致性筛选（默认开启）')
    parser.add_argument('--no_rotation_consensus', dest='rotation_consensus',
                        action='store_false', help='关闭旋转一致性筛选')
    parser.add_argument('--translation_parallel_threshold_deg', type=float, default=5.0,
                        help='射线夹角小于该阈值时跳过射线距离计算（度）')
    parser.add_argument('--translation_consensus', dest='translation_consensus',
                        action='store_true', default=True,
                        help='开启平移射线一致性筛选（默认开启）')
    parser.add_argument('--no_translation_consensus', dest='translation_consensus',
                        action='store_false', help='关闭平移射线一致性筛选')
    parser.add_argument('--query_image', nargs='+')
    parser.add_argument('--pose_details', action='store_true',
                        help='在 output_dir/pose_details 下保存逐 query 的筛选详情')
    parser.add_argument('--amp', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.topk < 1:
        parser.error('--topk must be positive')
    if not np.isfinite(args.rotation_threshold_deg) or args.rotation_threshold_deg < 0:
        parser.error('--rotation_threshold_deg must be finite and non-negative')
    if (not np.isfinite(args.translation_parallel_threshold_deg)
            or not 0.0 <= args.translation_parallel_threshold_deg <= 90.0):
        parser.error('--translation_parallel_threshold_deg must be between 0 and 90')
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'InLoc')
    os.makedirs(args.output_dir, exist_ok=True)

    root = Path(args.dataset_dir)
    # 每个 query 的候选保持文件中的排名顺序，topk 只截取前 K 个。
    pairs = read_pairs(args.pairs)
    if args.query_image:
        requested = set(args.query_image)
        pairs = {q: dbs for q, dbs in pairs.items()
                 if q in requested or q.removeprefix('iphone7/') in requested}
    if not pairs:
        raise ValueError('No query pairs found in {}'.format(args.pairs))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_reloc3r_relpose_model(model_args=args.img_reso, device=device)
    visloc = Reloc3rVisloc()
    # 图像解码和 scan pose 恢复都较昂贵，按路径缓存，避免同一 map 重复处理。
    cache = {}
    pose_cache = {}

    def image(path):
        """加载图像并按 Reloc3r 要求缩放/补边，结果缓存为 CPU 张量。"""
        path = str(path)
        if path not in cache:
            item = load_images([path], size=int(args.img_reso), verbose=False)[0]
            _, _, h, w = item['img'].shape
            # Reloc3r 只接受横屏方向。横屏和正方形图像保持 load_images 的
            # 输出，不做额外缩放或补边；只有竖屏图像才按参数转换成正方形。
            if w < h:
                side = h
                if args.image_resize == 'stretch':
                    # stretch：将竖屏图像直接拉伸到 h x h，可能改变图像比例。
                    item['img'] = torch.nn.functional.interpolate(
                        item['img'], size=(side, side), mode='bilinear',
                        align_corners=False)
                else:
                    # pad：保持竖屏图像比例，在左右对称补边到 h x h。
                    pad_left = (side - w) // 2
                    pad_right = side - w - pad_left
                    item['img'] = torch.nn.functional.pad(
                        item['img'], (pad_left, pad_right, 0, 0), value=-1.0)
                item['true_shape'] = np.asarray([[side, side]], dtype=np.int32)
            cache[path] = item
        item = cache[path]
        return {'img': item['img'].to(device), 'true_shape': torch.as_tensor(item['true_shape']).to(device)}

    def save_pose_details(query, pair_details, rotation_rows, medoid_name,
                          failure_reason=None, query_c2w=None, query_w2c=None):
        """保存 query 的相对位姿、绝对旋转一致性及最终位姿诊断。"""
        if not args.pose_details:
            return
        details_dir = Path(args.output_dir) / 'pose_details'
        details_dir.mkdir(parents=True, exist_ok=True)
        safe_name = query.replace('\\', '_').replace('/', '_')
        details_path = details_dir / (safe_name + '.txt')
        with details_path.open('w', encoding='utf-8') as f:
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
        """按 reloc.py 格式写出射线夹角、距离矩阵和鲁棒分数。"""
        if not args.pose_details:
            return
        details_dir = Path(args.output_dir) / 'pose_details'
        details_dir.mkdir(parents=True, exist_ok=True)
        safe_name = query.replace('\\', '_').replace('/', '_')
        details_path = details_dir / (safe_name + '.trans.txt')
        with details_path.open('w', encoding='utf-8') as f:
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
    for query, database_names in tqdm(pairs.items(), desc='Localizing InLoc'):
        query_path = root / Path(query.replace('\\', '/'))
        if not query_path.is_file():
            print('Skip missing query:', query_path)
            continue
        q_img = image(query_path)
        poses_db, poses_q2d, rotations, used_names = [], [], [], []
        pair_details = []
        for db_name in database_names[:args.topk]:
            db_path = root / Path(db_name.replace('\\', '/'))
            if not db_path.is_file():
                print('Skip missing map image:', db_path)
                continue
            try:
                if db_name not in pose_cache:
                    pose_cache[db_name] = cutout_camera_to_world(root, db_name)
                T_db = pose_cache[db_name]
            except (ValueError, FileNotFoundError) as exc:
                print('Skip {}: {}'.format(db_name, exc))
                continue
            # view1=数据库 cutout，view2=query；Reloc3r 返回 query 到数据库相机的变换。
            pred = to_numpy(inference_relpose([image(db_path), q_img], model, device,
                                              use_amp=bool(args.amp))[0])
            norm = np.linalg.norm(pred[:3, 3])
            if norm < 1e-8 or not np.isfinite(pred).all():
                pair_details.append({'name': db_name, 'pose': pred.copy(),
                                     'absolute_rotation': None, 'valid': False})
                continue
            pred[:3, 3] /= norm
            absolute_rotation = (T_db @ pred)[:3, :3]
            pair_details.append({'name': db_name, 'pose': pred.copy(),
                                 'absolute_rotation': absolute_rotation.copy(), 'valid': True})
            poses_db.append(T_db)
            poses_q2d.append(pred)
            rotations.append(absolute_rotation)
            used_names.append(db_name)
        if len(poses_db) < 2:
            print('Skip {}: fewer than two valid candidates'.format(query))
            save_pose_details(query, pair_details, [], None,
                              'fewer than two valid candidates')
            continue
        # 先比较每个候选推得的 query 绝对旋转，只留下 medoid 邻域内的候选。
        distances, medoid, rotation_keep = rotation_consensus(
            rotations, args.rotation_threshold_deg)
        medoid_name = used_names[medoid]
        rotation_rows = [(name, float(distances[medoid, i]), bool(rotation_keep[i]))
                         for i, name in enumerate(used_names)]
        if not args.rotation_consensus:
            rotation_keep = np.ones(len(used_names), dtype=bool)
            rotation_rows = [(name, None, True) for name in used_names]
        if np.count_nonzero(rotation_keep) < 2:
            medoid_distances = np.sort(distances[medoid])
            print('Skip {}: rotation consensus kept {}/{} (threshold {:.1f} deg; '
                  'nearest distances {})'.format(
                      query, int(np.count_nonzero(rotation_keep)), len(rotation_keep),
                      args.rotation_threshold_deg,
                      np.array2string(medoid_distances[:5], precision=1)))
            save_pose_details(query, pair_details, rotation_rows, medoid_name,
                              'rotation consensus retained fewer than two candidates')
            continue
        poses_db = [p for p, k in zip(poses_db, rotation_keep) if k]
        poses_q2d = [p for p, k in zip(poses_q2d, rotation_keep) if k]
        used_names = [n for n, k in zip(used_names, rotation_keep) if k]
        if len(poses_db) < 2:
            print('Skip {}: rotation consensus retained fewer than two candidates'.format(query))
            save_pose_details(query, pair_details, rotation_rows, medoid_name,
                              'rotation consensus retained fewer than two candidates')
            continue
        # 每个 map 位姿给出射线起点；相对平移方向经 map 朝向转到全局坐标系。
        centers = np.asarray([p[:3, 3] for p in poses_db])
        directions = np.asarray([p[:3, :3] @ q[:3, 3] for p, q in zip(poses_db, poses_q2d)])
        translation_keep, translation_details = translation_ray_consensus(
            centers, directions, used_names,
            args.translation_parallel_threshold_deg)
        if not args.translation_consensus:
            translation_keep = np.ones(len(used_names), dtype=bool)
        save_translation_details(query, used_names, translation_keep, translation_details)
        poses_db = [p for p, k in zip(poses_db, translation_keep) if k]
        poses_q2d = [p for p, k in zip(poses_q2d, translation_keep) if k]
        if len(poses_db) < 2:
            print('Skip {}: translation consensus retained fewer than two candidates'.format(query))
            save_pose_details(query, pair_details, rotation_rows, medoid_name,
                              'translation consensus retained fewer than two candidates')
            continue
        # 对通过两级一致性筛选的候选执行旋转平均和射线交会。
        try:
            pose_c2w = visloc.motion_averaging(poses_db, poses_q2d)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            print('Skip {}: {}'.format(query, exc))
            save_pose_details(query, pair_details, rotation_rows, medoid_name, str(exc))
            continue
        pose_w2c = np.linalg.inv(pose_c2w)
        save_pose_details(query, pair_details, rotation_rows, medoid_name,
                          query_c2w=pose_c2w, query_w2c=pose_w2c)
        results.append((query, pose_w2c, medoid_name, len(poses_db)))

    if not results:
        raise RuntimeError('No InLoc queries could be localized.')
    out_path = Path(args.output_dir) / 'query_poses.txt'
    with out_path.open('w', encoding='utf-8') as f:
        f.write('# Estimated query poses (world-to-camera; inferred DUC scan frame)\n')
        f.write('# query QW QX QY QZ TX TY TZ retained_candidates\n')
        for name, T, _, count in results:
            qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write('{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {}\n'.format(
                name, qw, qx, qy, qz, *T[:3, 3], count))
    print('Localized {} queries; poses saved to {}'.format(len(results), out_path))


if __name__ == '__main__':
    main()
