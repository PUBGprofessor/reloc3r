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
    names = sorted(k for k in h5py.File(path, 'r').keys())
    descs = []
    with h5py.File(path, 'r') as f:
        for name in names:
            descs.append(np.asarray(f[name]['global_descriptor'], dtype=np.float32))
    descs = np.stack(descs, axis=0) if descs else np.zeros((0, 4096), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descs = descs / norms
    return names, descs


def rotation_error_deg(R_est, R_gt):
    """Geodesic angle (degrees) between two rotation matrices."""
    R_err = R_est.T @ R_gt
    cos = (np.trace(R_err) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Reloc3r visual localization on Indoor (map/query).')
    parser.add_argument('--map_dir', type=str, default=r'D:\Datasets\Indoor_map')
    parser.add_argument('--query_dir', type=str, default=r'D:\Datasets\Indoor_query')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='default: <repo>/output/Indoor')
    parser.add_argument('--img_reso', type=str, default='512', choices=['224', '512'])
    parser.add_argument('--topk', type=int, default=5,
                        help='number of db images per query for motion averaging')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='use Automatic Mixed Precision (fp16)')
    args = parser.parse_args()

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

    # ---- NetVLAD descriptors -------------------------------------------
    print('Loading NetVLAD global descriptors...')
    db_names, db_descs_all = load_netvlad_h5(
        os.path.join(args.map_dir, 'hloc_outputs', 'global-feats-netvlad.h5'))
    q_names, q_descs = load_netvlad_h5(
        os.path.join(args.query_dir, 'hloc_outputs', 'global-feats-netvlad.h5'))
    print('  db descriptors: {}, query descriptors: {}'.format(len(db_names), len(q_names)))

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
            image_cache[img_path] = d
        return d

    def to_device(img_dict):
        return {'img': img_dict['img'].to(device),
                'true_shape': torch.as_tensor(img_dict['true_shape']).to(device)}

    # ---- localize every query ------------------------------------------
    results = []
    for qi, q_name in enumerate(tqdm(q_names, desc='Localizing')):
        if q_name not in query_gt:
            print('  skip {} (no GT pose)'.format(q_name))
            continue
        q_path = os.path.join(query_images_dir, q_name)
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
        for db_name in top_names:
            db_path = os.path.join(map_images_dir, db_name)
            if not os.path.exists(db_path):
                continue
            db_img = to_device(get_image(db_path))

            # view1 = db, view2 = query -> pose2to1 = query -> db
            batch = [db_img, q_img]
            pose_q2d = to_numpy(inference_relpose(batch, reloc3r_relpose, device,
                                                  use_amp=bool(args.amp))[0])
            # Reloc3r is scale-free: keep only the direction of the translation.
            t = pose_q2d[:3, 3]
            norm = float(np.linalg.norm(t))
            if norm > 1e-8:
                pose_q2d[:3, 3] = t / norm

            poses_db_c2w.append(np.linalg.inv(map_gt[db_name]))
            poses_q2d.append(pose_q2d)

        if len(poses_db_c2w) < 2:
            print('  skip {} (not enough valid db pairs)'.format(q_name))
            continue

        query_c2w = visloc.motion_averaging(poses_db_c2w, poses_q2d)
        query_w2c = np.linalg.inv(query_c2w)

        results.append(dict(name=q_name, w2c=query_w2c, c2w=query_c2w,
                            top_names=top_names, top_sims=sims[order]))

    if len(results) == 0:
        print('No query localized, aborting.')
        return

    # ---- save estimated poses (w2c, COLMAP style) -----------------------
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
        gt_c2w = np.linalg.inv(query_gt[r['name']])
        R_est, C_est = r['c2w'][:3, :3], r['c2w'][:3, 3]
        R_gt, C_gt = gt_c2w[:3, :3], gt_c2w[:3, 3]
        terr = float(np.linalg.norm(C_est - C_gt))
        rerr = rotation_error_deg(R_est, R_gt)
        terrs.append(terr)
        rerrs.append(rerr)
        detail_lines.append('{:20s}  t={:8.4f} m  r={:7.3f} deg  top1={}'.format(
            r['name'], terr, rerr, r['top_names'][0]))

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

    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(report + '\n')
    print('Evaluation saved to {}'.format(results_path))


if __name__ == '__main__':
    main()
