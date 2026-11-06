"""Interactive 3D visualization of multi-reference Reloc3r localization."""

import argparse
import os
import re

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def read_colmap_images_txt(path):
    """Return image-name -> world-to-camera matrix from COLMAP images.txt."""
    poses = {}
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                translation = np.asarray(list(map(float, parts[5:8])))
            except ValueError:
                i += 1
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose[:3, 3] = translation
            poses[parts[9]] = pose
            i += 2
        else:
            i += 1
    return poses


def _parse_matrix(text, description):
    values = np.fromstring(text.replace('[', ' ').replace(']', ' '), sep=' ')
    if values.size != 16:
        raise ValueError('Could not parse {} as a 4x4 matrix'.format(description))
    return values.reshape(4, 4)


def read_pose_details(path):
    """Parse the text written by ``reloc.py --pose_details``."""
    with open(path, 'r') as f:
        content = f.read()
    query_match = re.search(r'^Query image:\s*([^\r\n]+)$', content, re.MULTILINE)
    c2w_match = re.search(
        r'Final camera-to-world pose:\s*\r?\n(\[\[.*?\]\])', content, re.DOTALL)
    if query_match is None or c2w_match is None:
        raise ValueError('Details file is missing the query name or final c2w pose: {}'.format(path))
    query_name = query_match.group(1).strip()
    estimated_c2w = _parse_matrix(c2w_match.group(1), 'final camera-to-world pose')

    # Newer details files include the rotation-consensus decision for every
    # database candidate.  Keep this separate from the pose parser so the
    # existing relative-pose format remains unchanged.
    consensus = {}
    consensus_pattern = re.compile(
        r'^\s{2}(.*?)\s+/\s+(?:invalid|[\d.eE+-]+\s+deg)\s+/\s+(yes|no)\s*$',
        re.MULTILINE)
    for match in consensus_pattern.finditer(content):
        consensus[match.group(1).strip()] = match.group(2) == 'yes'

    pairs = []
    pair_pattern = re.compile(
        r'Database image:\s*([^\r\n]+)\r?\n'
        r'Retrieval similarity:\s*([\d.eE+-]+)\r?\n'
        r'(?:Translation direction error:\s*(?:[\d.eE+-]+ deg|N/A)\r?\n'
        r'Rotation error:\s*(?:[\d.eE+-]+ deg|N/A)\r?\n)?'
        r'(\[\[.*?\]\])', re.DOTALL)
    for match in pair_pattern.finditer(content):
        db_name = match.group(1).strip()
        similarity = float(match.group(2))
        relative_pose = _parse_matrix(match.group(3), 'relative pose for {}'.format(db_name))
        # ``None`` keeps old pose-details files usable when no consensus
        # section is present.
        pairs.append((db_name, similarity, relative_pose, consensus.get(db_name)))
    if not pairs:
        raise ValueError('No database relative poses found in {}'.format(path))
    return query_name, estimated_c2w, pairs


def camera_geometry(c2w, width, height, intrinsics, color, scale):
    """Create the same Open3D camera cone used by visualization.py."""
    geometry = o3d.geometry.LineSet.create_camera_visualization(
        width, height, intrinsics, np.linalg.inv(c2w), scale=scale)
    geometry.paint_uniform_color(color)
    return geometry


def line_geometry(start, end, color):
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(np.asarray([start, end], dtype=np.float64))
    geometry.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    geometry.colors = o3d.utility.Vector3dVector(np.asarray([color], dtype=np.float64))
    return geometry


def main():
    parser = argparse.ArgumentParser(description='Interactive multi-view Reloc3r visualization.')
    parser.add_argument('--map_dir', required=True)
    parser.add_argument('--query_dir', required=True)
    parser.add_argument('--details_path', required=True)
    parser.add_argument('--ray_length', type=float, default=80.0,
                        help='display length of each scale-free ray')
    parser.add_argument('--camera_scale', type=float, default=0.8,
                        help='display scale of each camera cone')
    args = parser.parse_args()

    query_name, estimated_c2w, pairs = read_pose_details(args.details_path)
    map_poses = read_colmap_images_txt(os.path.join(args.map_dir, 'sparse', '0', 'images.txt'))
    query_poses = read_colmap_images_txt(os.path.join(args.query_dir, 'sparse', '0', 'images.txt'))
    if query_name not in query_poses:
        raise KeyError('Query image {!r} is not present in query images.txt'.format(query_name))

    query_gt_c2w = np.linalg.inv(query_poses[query_name])
    image_width, image_height = 640, 480
    intrinsics = np.array([[525.0, 0.0, image_width / 2.0],
                           [0.0, 525.0, image_height / 2.0],
                           [0.0, 0.0, 1.0]])
    query_gt_color = (0.15, 0.75, 0.25)
    estimate_color = (0.90, 0.15, 0.12)
    db_colors = [(0.12, 0.40, 0.95), (0.95, 0.55, 0.08), (0.65, 0.20, 0.80),
                 (0.05, 0.65, 0.65), (0.85, 0.25, 0.50), (0.35, 0.65, 0.12)]

    geometries = []
    labels = []
    geometries.append(camera_geometry(query_gt_c2w, image_width, image_height,
                                      intrinsics, query_gt_color, args.camera_scale))
    labels.append((query_gt_c2w[:3, 3], 'Query GT'))
    geometries.append(camera_geometry(estimated_c2w, image_width, image_height,
                                      intrinsics, estimate_color, args.camera_scale))
    labels.append((estimated_c2w[:3, 3], 'Query estimated'))

    for index, (db_name, similarity, relative_pose, retained) in enumerate(pairs):
        if db_name not in map_poses:
            raise KeyError('Database image {!r} is not present in map images.txt'.format(db_name))
        db_c2w = np.linalg.inv(map_poses[db_name])
        db_color = db_colors[index % len(db_colors)]
        db_center = db_c2w[:3, 3]
        geometries.append(camera_geometry(db_c2w, image_width, image_height,
                                          intrinsics, db_color, args.camera_scale))
        if retained is True:
            selection_mark = ' [selected]'
        elif retained is False:
            selection_mark = ' [rejected]'
        else:
            selection_mark = ''
        labels.append((db_center, '{} ({:.3f}){}'.format(
            db_name, similarity, selection_mark)))

        # Relative translation is scale-free. Rotate its direction to world
        # coordinates and draw the ray from the database camera center only.
        direction = db_c2w[:3, :3] @ relative_pose[:3, 3]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            raise ValueError('Relative translation for {!r} has near-zero length'.format(db_name))
        direction /= direction_norm
        # 射线
        geometries.append(line_geometry(db_center, db_center + args.ray_length * direction, db_color))
        # 射线反向，长度减半
        # geometries.append(line_geometry(db_center, db_center - args.ray_length * direction / 2, db_color))
        

    # Basic Open3D window: drag to rotate and scroll to zoom; no axes/panel.
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    viewer = o3d.visualization.O3DVisualizer(
        'Reloc3r multi-view localization: {}'.format(query_name), 1280, 800)
    viewer.show_settings = False
    for index, geometry in enumerate(geometries):
        viewer.add_geometry('geometry_{:03d}'.format(index), geometry)
    for position, text in labels:
        viewer.add_3d_label(position, text)
    viewer.reset_camera_to_default()
    app.add_window(viewer)
    app.run()


if __name__ == '__main__':
    main()
