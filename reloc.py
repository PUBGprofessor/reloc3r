"""Visual localization with Reloc3r on the Indoor dataset.

The Indoor_map / Indoor_query datasets are two halves of a single COLMAP
reconstruction (so they share one world coordinate system, w2c format).
Reloc3r is scale-free, so we cannot directly regress a metric absolute pose
from one query<->db pair. Instead, for every query image we:

  1. retrieve the top-K database images using the pre-extracted NetVLAD
     global descriptors (`hloc_outputs/global-feats-netvlad.h5`);
  2. run Reloc3r between the query and each of those K db images to get K
     scale-free relative poses;
  3. fuse them with rotation averaging + camera-center triangulation
     (the "ray intersection" step) to obtain the absolute query pose.

Ground-truth poses are read from COLMAP `sparse/0/images.txt` (w2c) and used
only for evaluation.

Run (inside the `dl` conda environment):
    conda activate dl
    python reloc.py
"""

import os
import argparse
import numpy as np
import h5py
import torch

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from reloc3r.utils.image import load_images
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.device import to_numpy


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------
def read_colmap_images_txt(path):
    """Parse a COLMAP `images.txt` (text format).

    Returns a dict {image_name: T_w2c (4x4 float64)} where T_w2c maps world
    points into the camera frame (COLMAP convention: R * X_world + t).
    """
    poses = {}
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line or line.startswith('#'):
            i += 1
            continue

        parts = line.split()
        # image line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) >= 10:
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
            except ValueError:
                i += 1
                continue
            name = parts[9]
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot
            T[:3, 3] = [tx, ty, tz]
            poses[name] = T
            # COLMAP stores two lines per image; skip the (possibly empty)
            # points2D line that follows.
            i += 2
        else:
            i += 1
    return poses


def load_netvlad_h5(path):
    """Load hloc NetVLAD global descriptors from an .h5 file.

    Each top-level key is an image name (e.g. 'frame00002.png') mapping to a
    group with a 'global_descriptor' dataset. Returns (names, descs) where
    descs has shape (N, 4096) and is L2-normalized float32.
    """
    with h5py.File(path, 'r') as f:
        descriptors = {}

        def collect_descriptors(name, item):
            if isinstance(item, h5py.Group) and 'global_descriptor' in item:
                descriptors[name] = np.asarray(item['global_descriptor'], dtype=np.float32)

        # HLoc may store image names with directory prefixes (e.g. db/1003.jpg),
        # which appear as nested groups in HDF5 rather than top-level keys.
        f.visititems(collect_descriptors)

    names = sorted(descriptors)
    descs = [descriptors[name] for name in names]
    descs = np.stack(descs, axis=0) if descs else np.zeros((0, 4096), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descs = descs / norms
    return names, descs


def resolve_image_path(dataset_dir, images_dir, image_name):
    """Resolve both flat image names and names containing subdirectory prefixes."""
    candidates = [os.path.join(images_dir, image_name), os.path.join(dataset_dir, image_name)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def rotation_error_deg(R_est, R_gt):
    """Geodesic angle (degrees) between two rotation matrices."""
    R_err = R_est.T @ R_gt
    cos = (np.trace(R_err) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def translation_direction_error_deg(t_est, t_gt):
    """Angular error between translation directions, independent of scale."""
    est_norm = np.linalg.norm(t_est)
    gt_norm = np.linalg.norm(t_gt)
    if est_norm < 1e-8 or gt_norm < 1e-8:
        return None
    cosine = np.dot(t_est, t_gt) / (est_norm * gt_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def rotation_consensus(rotations, threshold_deg=15.0):
    """Return pairwise angular distances, the medoid, and its inlier mask."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError('Expected candidate rotations with shape (N, 3, 3).')
    if len(rotations) == 0 or not np.isfinite(rotations).all():
        raise ValueError('Candidate rotations are empty or contain NaN/Inf.')

    distances = np.zeros((len(rotations), len(rotations)), dtype=np.float64)
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            relative = rotations[i].T @ rotations[j]
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cosine))
            distances[i, j] = distances[j, i] = angle_deg

    medoid_idx = int(np.argmin(distances.sum(axis=1)))
    inlier_mask = distances[medoid_idx] <= float(threshold_deg)
    return distances, medoid_idx, inlier_mask


def translation_ray_consensus(centers, directions, candidate_names,
                              parallel_threshold_deg=5.0):
    """Filter ray outliers using robust statistics of pairwise half-ray distances."""
    centers = np.asarray(centers, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    count = len(centers)
    if centers.shape != (count, 3) or directions.shape != (count, 3):
        raise ValueError('Ray centers and directions must have shape (N, 3).')
    if len(candidate_names) != count:
        raise ValueError('Ray candidate names must match the candidate count.')
    if not np.isfinite(centers).all() or not np.isfinite(directions).all():
        raise ValueError('Ray centers or directions contain NaN/Inf.')
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError('Ray direction has near-zero length.')
    directions = directions / norms[:, None]

    distances = np.full((count, count), np.nan, dtype=np.float64)
    angles = np.full((count, count), np.nan, dtype=np.float64)
    skipped_pairs = []
    for i in range(count):
        for j in range(i + 1, count):
            dot = float(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0))))
            angles[i, j] = angles[j, i] = angle_deg
            if angle_deg < parallel_threshold_deg:
                skipped_pairs.append((i, j, angle_deg))
                continue

            # Minimize |(C_i + s*d_i) - (C_j + t*d_j)| with s,t >= 0.
            # The unconstrained solution is valid when both parameters are
            # forward; otherwise the constrained minimum lies on a boundary.
            w = centers[i] - centers[j]
            b = float(np.dot(directions[i], directions[j]))
            rhs_i = -float(np.dot(directions[i], w))
            rhs_j = float(np.dot(directions[j], w))
            denom = 1.0 - b * b
            candidates = [(0.0, 0.0)]
            if denom > 1e-12:
                s = (rhs_i + b * rhs_j) / denom
                t = (rhs_j + b * rhs_i) / denom
                if s >= 0.0 and t >= 0.0:
                    candidates.append((s, t))
            # Boundary minima: s=0, t=0, and their shared endpoint.
            candidates.append((0.0, max(0.0, rhs_j)))
            candidates.append((max(0.0, rhs_i), 0.0))
            distances[i, j] = distances[j, i] = min(
                np.linalg.norm(w + s * directions[i] - t * directions[j])
                for s, t in candidates)

    min_comparisons = max(2, int(np.ceil(0.5 * (count - 1))))
    scores = np.full(count, np.nan, dtype=np.float64)
    valid_counts = np.zeros(count, dtype=np.int32)
    for i in range(count):
        valid = np.isfinite(distances[i])
        valid_counts[i] = int(valid.sum())
        if valid_counts[i] >= min_comparisons:
            scores[i] = float(np.median(distances[i, valid]))

    score_valid = np.isfinite(scores)
    inlier_mask = np.ones(count, dtype=bool)
    median_score = mad = threshold = float('nan')
    outlier_indices = []
    if int(score_valid.sum()) >= 3:
        median_score = float(np.median(scores[score_valid]))
        mad = float(np.median(np.abs(scores[score_valid] - median_score)))
        sigma = 1.4826 * mad
        threshold = max(2.0 * median_score, median_score + 4.0 * sigma)
        inlier_mask[score_valid] = scores[score_valid] <= threshold
        outlier_indices = np.flatnonzero(~inlier_mask).tolist()
        min_remaining = max(2, int(np.ceil(0.5 * count)))
        if count - len(outlier_indices) < min_remaining:
            inlier_mask[:] = True
            outlier_indices = []
    else:
        sigma = float('nan')

    details = {
        'angles': angles,
        'distances': distances,
        'skipped_pairs': skipped_pairs,
        'valid_counts': valid_counts,
        'scores': scores,
        'min_comparisons': min_comparisons,
        'median': median_score,
        'mad': mad,
        'sigma': sigma,
        'threshold': threshold,
        'outlier_indices': outlier_indices,
    }
    return inlier_mask, details


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Reloc3r visual localization on Indoor (map/query).')
    parser.add_argument('--map_dir', type=str, default=r'D:\Datasets\Indoor_map_51')
    parser.add_argument('--query_dir', type=str, default=r'D:\Datasets\Indoor_query_38_38')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='default: <repo>/output/Indoor')
    parser.add_argument('--img_reso', type=str, default='512', choices=['224', '512'])
    parser.add_argument('--image_resize', choices=['stretch', 'pad'], default='pad',
                        help='square input handling: stretch images (default) or pad while preserving aspect ratio')
    parser.add_argument('--topk', type=int, default=5,
                        help='number of db images per query for motion averaging')
    parser.add_argument('--rotation_threshold_deg', type=float, default=15.0,
                        help='maximum rotation distance from the medoid, in degrees')
    parser.add_argument('--rotation_consensus', dest='rotation_consensus',
                        action='store_true', default=True,
                        help='enable rotation-consensus filtering (default: enabled)')
    parser.add_argument('--no_rotation_consensus', dest='rotation_consensus',
                        action='store_false', help='disable rotation-consensus filtering')
    parser.add_argument('--translation_parallel_threshold_deg', type=float, default=5.0,
                        help='skip ray-pair distances below this acute angle, in degrees')
    parser.add_argument('--translation_consensus', dest='translation_consensus',
                        action='store_true', default=True,
                        help='enable translation-consensus filtering (default: enabled)')
    parser.add_argument('--no_translation_consensus', dest='translation_consensus',
                        action='store_false', help='disable translation-consensus filtering')
    parser.add_argument('--query_image', type=str, nargs='+', default=None,
                        help='localize only these query image names (as stored in COLMAP/H5)')
    parser.add_argument('--pose_details', action='store_true',
                        help='save per-query relative poses and final pose under output_dir/pose_details')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='use Automatic Mixed Precision (fp16)')
    args = parser.parse_args()
    if not np.isfinite(args.rotation_threshold_deg) or args.rotation_threshold_deg < 0:
        parser.error('--rotation_threshold_deg must be a finite non-negative number')
    if (not np.isfinite(args.translation_parallel_threshold_deg)
            or not 0.0 <= args.translation_parallel_threshold_deg <= 90.0):
        parser.error('--translation_parallel_threshold_deg must be between 0 and 90')

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'output', 'Indoor')
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    map_images_dir = os.path.join(args.map_dir, 'images')
    query_images_dir = os.path.join(args.query_dir, 'images')

    # ---- ground-truth poses (w2c) --------------------------------------
    print('Loading COLMAP ground-truth poses...')
    map_gt = read_colmap_images_txt(os.path.join(args.map_dir, 'sparse', '0', 'images.txt'))
    query_gt = read_colmap_images_txt(os.path.join(args.query_dir, 'sparse', '0', 'images.txt'))
    print('  map images: {}, query images: {}'.format(len(map_gt), len(query_gt)))

    # ---- Megaloc descriptors -------------------------------------------
    print('Loading Megaloc global descriptors...')
    db_names, db_descs_all = load_netvlad_h5(
        # os.path.join(args.map_dir, 'hloc_outputs', 'global-feats-netvlad.h5'))
        os.path.join(args.map_dir, 'hloc_outputs', 'global-feats-megaloc.h5'))
    q_names, q_descs = load_netvlad_h5(
        # os.path.join(args.query_dir, 'hloc_outputs', 'global-feats-netvlad.h5'))
        os.path.join(args.query_dir, 'hloc_outputs', 'global-feats-megaloc.h5'))
    print('  db descriptors: {}, query descriptors: {}'.format(len(db_names), len(q_names)))

    if args.query_image is not None:
        missing_queries = sorted(set(args.query_image) - set(q_names))
        if missing_queries:
            raise ValueError('Query images not found in the query descriptors: {}'.format(
                ', '.join(missing_queries)))

    # keep only db images that also have a pose and a descriptor
    keep = [i for i, n in enumerate(db_names) if n in map_gt]
    db_names = [db_names[i] for i in keep]
    db_descs = db_descs_all[keep]
    print('  usable db images: {}'.format(len(db_names)))

    # ---- Reloc3r model --------------------------------------------------
    print('Loading Reloc3r-{} model...'.format(args.img_reso))
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=args.img_reso, device=device)
    visloc = Reloc3rVisloc()

    # ---- image cache ----------------------------------------------------
    image_cache = {}

    def get_image(img_path):
        d = image_cache.get(img_path)
        if d is None:
            d = load_images([img_path], size=int(args.img_reso), verbose=False)[0]
            _, _, height, width = d['img'].shape
            if args.image_resize == 'stretch':
                # Reloc3r expects square input; resize directly to the model resolution.
                d['img'] = torch.nn.functional.interpolate(
                    d['img'], size=(int(args.img_reso), int(args.img_reso)),
                    mode='bilinear', align_corners=False)
                d['true_shape'] = np.asarray(
                    [[int(args.img_reso), int(args.img_reso)]], dtype=np.int32)
            else:
                # Pad symmetrically along the shorter dimension to preserve aspect ratio.
                side = max(height, width)
                pad_top = (side - height) // 2
                pad_bottom = side - height - pad_top
                pad_left = (side - width) // 2
                pad_right = side - width - pad_left
                if pad_top or pad_bottom or pad_left or pad_right:
                    d['img'] = torch.nn.functional.pad(
                        d['img'], (pad_left, pad_right, pad_top, pad_bottom), value=-1.0)
                d['true_shape'] = np.asarray([[side, side]], dtype=np.int32)
            image_cache[img_path] = d
        return d

    def to_device(img_dict):
        return {'img': img_dict['img'].to(device),
                'true_shape': torch.as_tensor(img_dict['true_shape']).to(device)}

    def save_pose_details(q_name, pair_details, candidate_count, medoid_name,
                          consensus_rows, avg_rotation, translation_success,
                          failure_reason, query_c2w=None, query_w2c=None):
        if not args.pose_details:
            return
        details_dir = os.path.join(args.output_dir, 'pose_details')
        os.makedirs(details_dir, exist_ok=True)
        details_name = q_name.replace('\\', '_').replace('/', '_')
        details_path = os.path.join(details_dir, details_name + '.txt')
        with open(details_path, 'w') as f:
            f.write('Query image: {}\n'.format(q_name))
            f.write('Raw candidate count: {}\n'.format(candidate_count))
            f.write('Rotation threshold: {:.6f} deg\n'.format(args.rotation_threshold_deg))
            f.write('Rotation medoid: {}\n'.format(medoid_name or 'N/A'))
            f.write('Rotation consensus (candidate / error / retained):\n')
            for name, error_deg, retained in consensus_rows:
                error_text = 'invalid' if error_deg is None else '{:.6f} deg'.format(error_deg)
                f.write('  {} / {} / {}\n'.format(
                    name, error_text, 'yes' if retained else 'no'))
            f.write('Retained candidate count: {}\n'.format(
                sum(1 for _, _, retained in consensus_rows if retained)))
            f.write('Relative poses (query-to-database; translation direction normalized):\n')
            for db_name, similarity, relative_pose, translation_error, pair_rotation_error, _ in pair_details:
                f.write('\nDatabase image: {}\n'.format(db_name))
                f.write('Retrieval similarity: {:.9f}\n'.format(similarity))
                if translation_error is None:
                    f.write('Translation direction error: N/A\n')
                else:
                    f.write('Translation direction error: {:.6f} deg\n'.format(
                        translation_error))
                if pair_rotation_error is None:
                    f.write('Rotation error: N/A\n')
                else:
                    f.write('Rotation error: {:.6f} deg\n'.format(pair_rotation_error))
                f.write('{}\n'.format(np.array2string(relative_pose, precision=9)))
            if avg_rotation is not None:
                f.write('\nFinal rotation average (camera-to-world):\n')
                f.write('{}\n'.format(np.array2string(avg_rotation, precision=9)))
            f.write('\nTranslation least-squares solve: {}\n'.format(
                'success' if translation_success else 'failed'))
            if failure_reason:
                f.write('Localization failure: {}\n'.format(failure_reason))
            if query_c2w is not None:
                f.write('\nFinal camera-to-world pose:\n')
                f.write('{}\n'.format(np.array2string(query_c2w, precision=9)))
            if query_w2c is not None:
                f.write('\nFinal world-to-camera pose:\n')
                f.write('{}\n'.format(np.array2string(query_w2c, precision=9)))

    def save_translation_details(q_name, names, keep_mask, details):
        details_dir = os.path.join(args.output_dir, 'pose_details')
        os.makedirs(details_dir, exist_ok=True)
        details_path = os.path.join(details_dir, q_name.replace('\\', '_').replace('/', '_')
                                    + '.trans.txt')
        with open(details_path, 'w') as f:
            f.write('Query image: {}\n'.format(q_name))
            f.write('Candidate count before translation filtering: {}\n'.format(len(names)))
            f.write('Parallel-pair threshold: {:.6f} deg\n'.format(
                args.translation_parallel_threshold_deg))
            f.write('Pairwise acute ray angles (deg; NaN means unavailable):\n')
            f.write('{}\n'.format(np.array2string(details['angles'], precision=6)))
            f.write('Skipped near-parallel pairs (< threshold):\n')
            for i, j, angle in details['skipped_pairs']:
                f.write('  {} <-> {}: {:.6f} deg\n'.format(names[i], names[j], angle))
            f.write('Pairwise half-ray distance matrix (NaN means invalid/skipped):\n')
            f.write('{}\n'.format(np.array2string(details['distances'], precision=6)))
            f.write('Minimum valid comparisons per candidate: {}\n'.format(
                details['min_comparisons']))
            f.write('Candidate scores (valid comparisons / median distance / status):\n')
            for i, name in enumerate(names):
                score = details['scores'][i]
                score_text = 'Insufficient Geometry' if not np.isfinite(score) else '{:.6f}'.format(score)
                status = 'retained' if keep_mask[i] else 'rejected'
                f.write('  {} / {} / {} / {}\n'.format(
                    name, int(details['valid_counts'][i]), score_text, status))
            f.write('m: {}\n'.format(details['median']))
            f.write('MAD: {}\n'.format(details['mad']))
            f.write('Robust sigma: {}\n'.format(details['sigma']))
            f.write('Final threshold T: {}\n'.format(details['threshold']))
            f.write('Retained candidates: {}\n'.format(
                ', '.join(name for name, keep in zip(names, keep_mask) if keep)))
            f.write('Rejected candidates: {}\n'.format(
                ', '.join(name for name, keep in zip(names, keep_mask) if not keep) or 'None'))
            f.write('Candidate count after translation filtering: {}\n'.format(
                int(np.count_nonzero(keep_mask))))

    # ---- localize every query ------------------------------------------
    results = []
    query_items = [(qi, name) for qi, name in enumerate(q_names)
                   if args.query_image is None or name in args.query_image]
    for qi, q_name in tqdm(query_items, desc='Localizing'):
        if q_name not in query_gt and args.query_image is None:
            print('  skip {} (no GT pose)'.format(q_name))
            continue
        q_path = resolve_image_path(args.query_dir, query_images_dir, q_name)
        if not os.path.exists(q_path):
            print('  skip {} (no image file)'.format(q_name))
            continue

        q_img = to_device(get_image(q_path))

        # 1) retrieval
        sims = db_descs @ q_descs[qi]          # cosine similarity (N_db,)
        order = np.argsort(-sims)[:args.topk]
        top_names = [db_names[i] for i in order]

        # 2) relative poses + 3) motion averaging (ray intersection)
        poses_db_c2w = []
        poses_q2d = []
        pair_details = []
        candidate_rotations = []
        candidate_names = []
        for db_idx in order:
            db_name = db_names[db_idx]
            db_path = resolve_image_path(args.map_dir, map_images_dir, db_name)
            if not os.path.exists(db_path):
                continue
            db_img = to_device(get_image(db_path))

            # view1 = db, view2 = query -> pose2to1 = query -> db
            batch = [db_img, q_img]
            pose_q2d = to_numpy(inference_relpose(batch, reloc3r_relpose, device,
                                                  use_amp=bool(args.amp))[0])
            pose_db_c2w = np.linalg.inv(map_gt[db_name])
            # Reloc3r is scale-free: keep only the direction of the translation.
            t = pose_q2d[:3, 3]
            norm = float(np.linalg.norm(t))
            if norm > 1e-8:
                pose_q2d[:3, 3] = t / norm

            translation_error = None
            rotation_error = None
            if q_name in query_gt:
                gt_q2d = map_gt[db_name] @ np.linalg.inv(query_gt[q_name])
                translation_error = translation_direction_error_deg(
                    pose_q2d[:3, 3], gt_q2d[:3, 3])
                rotation_error = rotation_error_deg(
                    pose_q2d[:3, :3], gt_q2d[:3, :3])

            candidate_rotation = (pose_db_c2w @ pose_q2d)[:3, :3]
            valid_candidate = (candidate_rotation.shape == (3, 3)
                               and np.isfinite(candidate_rotation).all()
                               and np.isfinite(pose_q2d).all())
            pair_details.append((db_name, float(sims[db_idx]), pose_q2d.copy(),
                                 translation_error, rotation_error,
                                 candidate_rotation.copy() if valid_candidate else None))
            if valid_candidate:
                candidate_names.append(db_name)
                candidate_rotations.append(candidate_rotation)
                poses_db_c2w.append(pose_db_c2w)
                poses_q2d.append(pose_q2d)

        raw_candidate_count = len(pair_details)
        consensus_rows = [(detail[0], None, False) for detail in pair_details]
        medoid_name = None
        avg_rotation = None
        translation_success = False
        failure_reason = None
        if len(candidate_rotations) == 0:
            failure_reason = 'no finite candidate rotations'
            print('  skip {} ({})'.format(q_name, failure_reason))
            save_pose_details(q_name, pair_details, raw_candidate_count, medoid_name,
                              consensus_rows, avg_rotation, translation_success,
                              failure_reason)
            continue
        # 第一步：按旋转 medoid 筛选候选；关闭时使用全 True 掩码保留全部候选。
        if args.rotation_consensus:
            distances, medoid_idx, inlier_mask = rotation_consensus(
                candidate_rotations, threshold_deg=args.rotation_threshold_deg)
            medoid_name = candidate_names[medoid_idx]
            valid_detail_indices = [i for i, detail in enumerate(pair_details)
                                    if detail[5] is not None]
            for local_idx, detail_idx in enumerate(valid_detail_indices):
                consensus_rows[detail_idx] = (
                    pair_details[detail_idx][0],
                    float(distances[medoid_idx, local_idx]),
                    bool(inlier_mask[local_idx]))
        else:
            inlier_mask = np.ones(len(candidate_names), dtype=bool)
            consensus_rows = [(detail[0], None, detail[5] is not None)
                              for detail in pair_details]

        # Apply the same consensus indices to every corresponding candidate input.
        poses_db_c2w = [pose for pose, keep in zip(poses_db_c2w, inlier_mask) if keep]
        poses_q2d = [pose for pose, keep in zip(poses_q2d, inlier_mask) if keep]
        candidate_names = [name for name, keep in zip(candidate_names, inlier_mask) if keep]
        retained_count = int(np.count_nonzero(inlier_mask))
        print('  {}: {} candidates, medoid={}, retained={}'.format(
            q_name, raw_candidate_count, medoid_name, retained_count))
        if retained_count < 2:
            failure_reason = 'rotation consensus retained fewer than 2 candidates'
            print('  skip {} ({})'.format(q_name, failure_reason))
            save_pose_details(q_name, pair_details, raw_candidate_count, medoid_name,
                              consensus_rows, avg_rotation, translation_success,
                              failure_reason)
            continue

        # 第二步：从传给平均算法的同一候选生成射线，并同步应用平移筛选掩码。
        before_translation_count = len(candidate_names)
        if args.translation_consensus:
            ray_centers = np.asarray([pose_db[:3, 3] for pose_db in poses_db_c2w],
                                     dtype=np.float64)
            ray_directions = np.asarray([
                pose_db[:3, :3] @ pose_q[:3, 3]
                for pose_db, pose_q in zip(poses_db_c2w, poses_q2d)], dtype=np.float64)
            translation_keep, translation_details = translation_ray_consensus(
                ray_centers, ray_directions, candidate_names,
                parallel_threshold_deg=args.translation_parallel_threshold_deg)
            save_translation_details(q_name, candidate_names, translation_keep,
                                     translation_details)
            poses_db_c2w = [pose for pose, keep in zip(poses_db_c2w, translation_keep) if keep]
            poses_q2d = [pose for pose, keep in zip(poses_q2d, translation_keep) if keep]
            candidate_names = [name for name, keep in zip(candidate_names, translation_keep) if keep]
        print('  {}: translation consistency {}/{} candidates{}'.format(
            q_name, len(candidate_names), before_translation_count,
            ' retained' if args.translation_consensus else ' (disabled)'))

        try:
            candidate_abs_rotations = np.asarray([
                (pose_db @ pose_q)[:3, :3]
                for pose_db, pose_q in zip(poses_db_c2w, poses_q2d)])
            avg_rotation = visloc.rotation_averaging(candidate_abs_rotations)
            query_c2w = visloc.motion_averaging(poses_db_c2w, poses_q2d)
            translation_success = True
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failure_reason = str(exc)
            print('  skip {} (motion averaging failed: {})'.format(q_name, failure_reason))
            save_pose_details(q_name, pair_details, raw_candidate_count, medoid_name,
                              consensus_rows, avg_rotation, translation_success,
                              failure_reason)
            continue

        query_w2c = np.linalg.inv(query_c2w)

        results.append(dict(name=q_name, w2c=query_w2c, c2w=query_c2w,
                            top_names=top_names, top_sims=sims[order],
                            pair_details=pair_details))

        if args.query_image is not None:
            print('\nQuery image:', q_name)
            print('Estimated pose (camera-to-world):\n', query_c2w)
            print('Estimated pose (world-to-camera):\n', query_w2c)

        save_pose_details(q_name, pair_details, raw_candidate_count, medoid_name,
                          consensus_rows, avg_rotation, translation_success,
                          failure_reason, query_c2w, query_w2c)

    if len(results) == 0:
        print('No query localized, aborting.')
        return

    # ---- save estimated poses (w2c, COLMAP style) -----------------------
    if args.query_image is None:
        pose_path = os.path.join(args.output_dir, 'query_poses.txt')
        with open(pose_path, 'w') as f:
            f.write('# Estimated query poses (world-to-camera, COLMAP convention)\n')
            f.write('# name QW QX QY QZ TX TY TZ\n')
            for r in results:
                T = r['w2c']
                qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
                tx, ty, tz = T[:3, 3]
                f.write('{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(
                    r['name'], qw, qx, qy, qz, tx, ty, tz))
        print('Poses saved to {}'.format(pose_path))

    # ---- evaluate against GT --------------------------------------------
    terrs, rerrs = [], []
    detail_lines = []
    for r in results:
        if r['name'] not in query_gt:
            continue
        gt_c2w = np.linalg.inv(query_gt[r['name']])
        R_est, C_est = r['c2w'][:3, :3], r['c2w'][:3, 3]
        R_gt, C_gt = gt_c2w[:3, :3], gt_c2w[:3, 3]
        terr = float(np.linalg.norm(C_est - C_gt))
        rerr = rotation_error_deg(R_est, R_gt)
        terrs.append(terr)
        rerrs.append(rerr)
        detail_lines.append('{:20s}  t={:8.4f} m  r={:7.3f} deg  top1={}'.format(
            r['name'], terr, rerr, r['top_names'][0]))

    if not terrs:
        print('Ground-truth pose unavailable; localization completed without pose-error evaluation.')
        return

    terrs = np.array(terrs)
    rerrs = np.array(rerrs)

    # recall at standard thresholds (translation [m], rotation [deg])
    thresholds = [(0.25, 2.0), (0.25, 5.0), (0.5, 5.0),
                  (0.5, 10.0), (1.0, 10.0), (5.0, 30.0)]

    summary = []
    summary.append('Visual localization evaluation (Reloc3r)')
    summary.append('=' * 70)
    summary.append('Number of localized queries: {}'.format(len(results)))
    summary.append('')
    summary.append('Per-query errors (camera-center distance / rotation):')
    summary.extend(detail_lines)
    summary.append('')
    summary.append('Aggregate metrics:')
    summary.append('  translation error: mean {:.4f} m, median {:.4f} m, max {:.4f} m'.format(
        float(np.mean(terrs)), float(np.median(terrs)), float(np.max(terrs))))
    summary.append('  rotation error   : mean {:.4f} deg, median {:.4f} deg, max {:.4f} deg'.format(
        float(np.mean(rerrs)), float(np.median(rerrs)), float(np.max(rerrs))))
    summary.append('  recall @ (t <= T m AND r <= R deg):')
    for t_thr, r_thr in thresholds:
        ok = np.sum((terrs <= t_thr) & (rerrs <= r_thr))
        summary.append('    ({:5.2f} m, {:5.1f} deg) : {:.4f}  ({}/{})'.format(
            t_thr, r_thr, ok / len(results), int(ok), len(results)))

    report = '\n'.join(summary)
    print('\n' + report)

    if args.query_image is None:
        results_path = os.path.join(args.output_dir, 'results.txt')
        with open(results_path, 'w') as f:
            f.write(report + '\n')
        print('Evaluation saved to {}'.format(results_path))


if __name__ == '__main__':
    main()
