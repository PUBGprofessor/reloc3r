"""Interactive visualization of InLoc database and estimated query poses.

InLoc does not provide a COLMAP ``images.txt`` file for its database cutouts.
Each cutout has an ``XYZcut`` MATLAB file containing the 3-D point seen at
each pixel, while the corresponding DUC scan has an alignment transform.
This module reconstructs the cutout camera pose from those two files and then
uses the same Open3D camera/ray rendering style as ``multi_visualization.py``.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

try:
    import open3d as o3d
except ImportError:  # Keep pose/data validation usable without the GUI dependency.
    o3d = None


def _parse_matrix(text, description):
    """Parse a 4x4 NumPy-style matrix from text."""
    values = np.fromstring(text.replace('[', ' ').replace(']', ' '), sep=' ')
    if values.size != 16:
        raise ValueError('Could not parse {} as a 4x4 matrix'.format(description))
    matrix = values.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError('{} contains non-finite values'.format(description))
    return matrix


def _normalise_name(name):
    """Use the slash convention stored in InLoc pair/detail files."""
    return name.strip().replace('\\', '/')


def _safe_query_filename(query_name):
    """Map an InLoc path such as ``iphone7/IMG_0748.JPG`` to its detail name."""
    return _normalise_name(query_name).replace('/', '_').replace('\\', '_')


def _read_consensus_rows(content):
    """Read rotation-consensus decisions from an InLoc details file."""
    rows = {}
    pattern = re.compile(
        r'^\s{2}(.*?)\s+/\s+(?:invalid|[\d.eE+-]+\s+deg)\s+/\s+(yes|no)\s*$',
        re.MULTILINE)
    for match in pattern.finditer(content):
        rows[_normalise_name(match.group(1))] = match.group(2) == 'yes'
    return rows


def _read_inloc_pairs(content, consensus):
    """Parse candidate blocks written by ``reloc_inloc.py``.

    InLoc details do not have the retrieval-similarity field used by the
    original SFM visualizer.  The score is therefore ``None`` for these
    candidates.  A small legacy parser is kept so the function remains useful
    with details files produced by older versions of the localization script.
    """
    pairs = []
    block_pattern = re.compile(
        r'Database image:\s*([^\r\n]+)\r?\n'
        r'(.*?)(?=\r?\nDatabase image:|\r?\nLocalization failure:|'
        r'\r?\nFinal camera-to-world pose:|\Z)',
        re.DOTALL)
    for match in block_pattern.finditer(content):
        name = _normalise_name(match.group(1))
        block = match.group(2)
        valid_match = re.search(r'^\s*valid:\s*(true|false)\s*$', block,
                                re.IGNORECASE | re.MULTILINE)
        valid = valid_match is None or valid_match.group(1).lower() == 'true'
        similarity_match = re.search(
            r'^\s*Retrieval similarity:\s*([\d.eE+-]+)\s*$', block,
            re.IGNORECASE | re.MULTILINE)
        similarity = (None if similarity_match is None
                      else float(similarity_match.group(1)))
        matrix_match = re.search(r'(\[\[.*?\]\])', block, re.DOTALL)
        if matrix_match is None:
            raise ValueError('Missing relative pose for database image {!r}'.format(name))
        relative_pose = _parse_matrix(matrix_match.group(1),
                                      'relative pose for {}'.format(name))
        pairs.append((name, similarity, relative_pose, consensus.get(name), valid))

    if pairs:
        return pairs

    # Older pose-details files contain a retrieval similarity before each
    # matrix.  Accept them as a convenience when an InLoc result was produced
    # by a mixed-version pipeline.
    legacy_pattern = re.compile(
        r'Database image:\s*([^\r\n]+)\r?\n'
        r'Retrieval similarity:\s*([\d.eE+-]+)\r?\n'
        r'(?:Translation direction error:\s*(?:[\d.eE+-]+ deg|N/A)\r?\n'
        r'Rotation error:\s*(?:[\d.eE+-]+ deg|N/A)\r?\n)?'
        r'(\[\[.*?\]\])', re.DOTALL)
    for match in legacy_pattern.finditer(content):
        name = _normalise_name(match.group(1))
        relative_pose = _parse_matrix(
            match.group(3), 'relative pose for {}'.format(name))
        pairs.append((name, float(match.group(2)), relative_pose,
                      consensus.get(name), True))
    return pairs


def read_pose_details(path):
    """Read one InLoc ``pose_details/<query>.txt`` file.

    Returns ``(query_name, estimated_c2w_or_none, pairs)``.  Each pair is
    ``(database_name, similarity_or_none, relative_pose, retained, valid)``.
    The final pose is optional because failed localizations can still have
    candidate diagnostics in their details file.
    """
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    query_match = re.search(r'^Query image:\s*([^\r\n]+)$', content, re.MULTILINE)
    if query_match is None:
        raise ValueError('Details file is missing the query name: {}'.format(path))
    query_name = _normalise_name(query_match.group(1))

    c2w_match = re.search(
        r'Final camera-to-world pose:\s*\r?\n(\[\[.*?\]\])',
        content, re.DOTALL)
    estimated_c2w = None
    if c2w_match is not None:
        estimated_c2w = _parse_matrix(c2w_match.group(1),
                                      'final camera-to-world pose')

    consensus = _read_consensus_rows(content)
    pairs = _read_inloc_pairs(content, consensus)
    if not pairs:
        raise ValueError('No database relative poses found in {}'.format(path))
    return query_name, estimated_c2w, pairs


def read_query_poses(path):
    """Read ``query_poses.txt`` and convert its world-to-camera poses to c2w."""
    poses = {}
    path = Path(path)
    if not path.is_file():
        return poses
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            if len(fields) < 8:
                raise ValueError('Invalid query pose line {} in {}'.format(line_no, path))
            name = _normalise_name(fields[0])
            try:
                qw, qx, qy, qz = map(float, fields[1:5])
                translation = np.asarray(list(map(float, fields[5:8])), dtype=np.float64)
            except ValueError as exc:
                raise ValueError('Invalid numeric query pose line {} in {}'.format(
                    line_no, path)) from exc
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            w2c[:3, 3] = translation
            poses[name] = np.linalg.inv(w2c)
    return poses


def _alignment_matrix(path):
    """Read the matrix under ``After general icp`` in a DUC transform file."""
    lines = path.read_text(encoding='utf-8').splitlines()
    marker = next((i for i, line in enumerate(lines)
                   if line.strip().lower().startswith('after general icp')), None)
    starts = [marker + 1] if marker is not None else []
    starts.extend(range(0, max(0, len(lines) - 3)))
    for start in starts:
        rows = []
        for line in lines[start:start + 4]:
            values = np.fromstring(line, sep=' ')
            if values.size < 4:
                rows = []
                break
            rows.append(values[:4])
        if len(rows) == 4:
            matrix = np.asarray(rows, dtype=np.float64)
            if np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
                return matrix
    raise ValueError('Could not find a valid 4x4 alignment matrix in {}'.format(path))


def scan_to_world(dataset_dir, database_name):
    """Return the DUC scan-to-world transform for an InLoc cutout."""
    parts = _normalise_name(database_name).split('/')
    if len(parts) < 4 or parts[0].lower() != 'cutouts':
        raise ValueError(
            'Expected cutouts/DUC1|DUC2/<scan>/<image>, got {}'.format(database_name))
    floor, scan_id, image_name = parts[1], parts[2], parts[3]
    alignment_dir = Path(dataset_dir) / '{}_alignment'.format(floor) / 'transformations'
    transform_path = alignment_dir / '{}_trans_{}.txt'.format(image_name[:3], scan_id)
    if not transform_path.is_file():
        raise FileNotFoundError('Missing scan alignment: {}'.format(transform_path))
    return _alignment_matrix(transform_path), image_name


def cutout_camera_to_world(dataset_dir, database_name, sample_count=80):
    """Recover an InLoc cutout camera-to-world pose from ``XYZcut``.

    ``XYZcut`` stores points in DUC scan coordinates.  The cutout has a 60
    degree horizontal field of view, so its pixel rays can be aligned to those
    points with a Kabsch/SVD solve.  The recovered camera center is the scan
    origin; the DUC alignment then places it in the global coordinate system.
    """
    T_world_scan, image_name = scan_to_world(dataset_dir, database_name)
    image_path = Path(dataset_dir) / Path(_normalise_name(database_name))
    mat_path = Path(str(image_path) + '.mat')
    if not mat_path.is_file():
        raise FileNotFoundError('Missing cutout XYZ data: {}'.format(mat_path))
    mat = loadmat(str(mat_path))
    if 'XYZcut' not in mat:
        raise ValueError('XYZcut array is missing in {}'.format(mat_path))
    xyz = np.asarray(mat['XYZcut'], dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError('Unexpected XYZcut shape {} in {}'.format(xyz.shape, mat_path))

    height, width, _ = xyz.shape
    step = max(1, min(height, width) // max(1, int(sample_count)))
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    points = xyz[0:height:step, 0:width:step].reshape(-1, 3)
    pixels_x = xs.reshape(-1)
    pixels_y = ys.reshape(-1)
    valid = np.isfinite(points).all(axis=1)
    points, pixels_x, pixels_y = points[valid], pixels_x[valid], pixels_y[valid]
    point_norms = np.linalg.norm(points, axis=1)
    valid = point_norms > 1e-6
    points = points[valid]
    pixels_x = pixels_x[valid]
    pixels_y = pixels_y[valid]
    point_norms = point_norms[valid]
    if len(points) < 20:
        raise ValueError('Too few valid XYZcut samples in {}'.format(mat_path))

    focal = width / (2.0 * np.tan(np.deg2rad(30.0)))
    rays_camera = np.column_stack((
        (pixels_x - (width - 1.0) / 2.0) / focal,
        (pixels_y - (height - 1.0) / 2.0) / focal,
        np.ones(len(points))))
    rays_camera /= np.linalg.norm(rays_camera, axis=1, keepdims=True)
    rays_scan = points / point_norms[:, None]

    covariance = rays_scan.T @ rays_camera
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    R_scan_camera = u @ correction @ vt

    T_world_camera = np.eye(4, dtype=np.float64)
    T_world_camera[:3, :3] = T_world_scan[:3, :3] @ R_scan_camera
    T_world_camera[:3, 3] = T_world_scan[:3, 3]
    return T_world_camera


def camera_geometry(c2w, width, height, intrinsics, color, scale):
    """Create the Open3D camera cone used by the original visualizer."""
    if o3d is None:
        raise ImportError(
            'Open3D is required to render the visualization. Install it from '
            'requirements.txt before running without --dry_run.')
    geometry = o3d.geometry.LineSet.create_camera_visualization(
        width, height, intrinsics, np.linalg.inv(c2w), scale=scale)
    geometry.paint_uniform_color(color)
    return geometry


def line_geometry(start, end, color):
    """Create a colored line from ``start`` to ``end``."""
    if o3d is None:
        raise ImportError(
            'Open3D is required to render the visualization. Install it from '
            'requirements.txt before running without --dry_run.')
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(
        np.asarray([start, end], dtype=np.float64))
    geometry.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    geometry.colors = o3d.utility.Vector3dVector(np.asarray([color], dtype=np.float64))
    return geometry


def _resolve_details_path(args, query_name):
    if args.details_path:
        return Path(args.details_path)
    if not query_name:
        raise ValueError('Provide --query_image when --details_path is omitted')
    return (Path(args.output_dir) / 'pose_details' /
            (_safe_query_filename(query_name) + '.txt'))


def _label_name(name):
    """Keep labels readable while retaining the DUC scan and cutout name."""
    return _normalise_name(name)


def main():
    parser = argparse.ArgumentParser(
        description='Interactive InLoc database/query pose visualization.')
    parser.add_argument('--dataset_dir', type=str, default=r'D:\Datasets\InLoc',
                        help='InLoc root containing cutouts and DUC alignments.')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\project\reloc3r\output\InLoc',
                        help='Directory containing query_poses.txt and pose_details/.')
    parser.add_argument('--details_path', type=str,
                        help='One pose_details/<query>.txt file. Defaults to the selected query.')
    parser.add_argument('--query_image', type=str,
                        help='InLoc query name, for example iphone7/IMG_0748.JPG.')
    parser.add_argument('--query_poses_path', type=str,
                        help='Estimated query pose file; defaults to output_dir/query_poses.txt.')
    parser.add_argument('--ray_length', type=float, default=80.0,
                        help='Display length of each scale-free relative-translation ray.')
    parser.add_argument('--camera_scale', type=float, default=0.8,
                        help='Display scale of each camera cone.')
    parser.add_argument('--xyz_samples', type=int, default=80,
                        help='Approximate XYZcut sampling grid size per image dimension.')
    parser.add_argument('--image_width', type=int, default=1600,
                        help='Width used by the displayed InLoc camera model.')
    parser.add_argument('--image_height', type=int, default=1200,
                        help='Height used by the displayed InLoc camera model.')
    parser.add_argument('--horizontal_fov_deg', type=float, default=60.0,
                        help='Horizontal field of view of an InLoc cutout.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Load and validate poses without opening an Open3D window.')
    args = parser.parse_args()

    if args.ray_length <= 0 or args.camera_scale <= 0:
        parser.error('--ray_length and --camera_scale must be positive')
    if args.xyz_samples < 1 or args.image_width < 1 or args.image_height < 1:
        parser.error('--xyz_samples, --image_width and --image_height must be positive')
    if not 0 < args.horizontal_fov_deg < 180:
        parser.error('--horizontal_fov_deg must be between 0 and 180')

    details_path = _resolve_details_path(args, args.query_image)
    if not details_path.is_file():
        raise FileNotFoundError('Missing pose details file: {}'.format(details_path))
    query_name, estimated_c2w, pairs = read_pose_details(details_path)
    if args.query_image and _normalise_name(args.query_image) != query_name:
        raise ValueError(
            '--query_image {!r} does not match details query {!r}'.format(
                args.query_image, query_name))

    details_estimated_c2w = estimated_c2w
    query_poses_path = (Path(args.query_poses_path) if args.query_poses_path else
                        Path(args.output_dir) / 'query_poses.txt')
    query_poses = read_query_poses(query_poses_path)
    if estimated_c2w is None:
        estimated_c2w = query_poses.get(query_name)
    if estimated_c2w is None:
        raise ValueError(
            'No estimated pose for {!r} in {} or {}'.format(
                query_name, details_path, query_poses_path))

    image_width, image_height = args.image_width, args.image_height
    focal = image_width / (2.0 * np.tan(np.deg2rad(args.horizontal_fov_deg / 2.0)))
    intrinsics = np.array([[focal, 0.0, (image_width - 1.0) / 2.0],
                           [0.0, focal, (image_height - 1.0) / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
    estimate_color = (0.90, 0.15, 0.12)
    db_colors = [(0.12, 0.40, 0.95), (0.95, 0.55, 0.08), (0.65, 0.20, 0.80),
                 (0.05, 0.65, 0.65), (0.85, 0.25, 0.50), (0.35, 0.65, 0.12)]

    # Keep ``--dry_run`` independent of the optional GUI dependency.  The
    # pose recovery and validation below are the expensive/data-specific part.
    if not args.dry_run and o3d is None:
        raise ImportError(
            'Open3D is required to render the visualization. Install it or use --dry_run.')
    geometries = []
    labels = [(estimated_c2w[:3, 3], 'Query estimated')]
    if not args.dry_run:
        geometries.append(camera_geometry(estimated_c2w, image_width, image_height,
                                          intrinsics, estimate_color, args.camera_scale))
    pose_cache = {}
    loaded_pairs = 0
    for index, (db_name, similarity, relative_pose, retained, valid) in enumerate(pairs):
        if not valid:
            continue
        if db_name not in pose_cache:
            pose_cache[db_name] = cutout_camera_to_world(
                args.dataset_dir, db_name, sample_count=args.xyz_samples)
        db_c2w = pose_cache[db_name]
        db_color = db_colors[index % len(db_colors)]
        db_center = db_c2w[:3, 3]
        if not args.dry_run:
            geometries.append(camera_geometry(db_c2w, image_width, image_height,
                                              intrinsics, db_color, args.camera_scale))
        if retained is True:
            selection_mark = ' [selected]'
        elif retained is False:
            selection_mark = ' [rejected]'
        else:
            selection_mark = ''
        score_text = '' if similarity is None else ' ({:.3f})'.format(similarity)
        labels.append((db_center, '{}{}{}'.format(
            _label_name(db_name), score_text, selection_mark)))

        direction = db_c2w[:3, :3] @ relative_pose[:3, 3]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10 or not np.isfinite(direction).all():
            raise ValueError('Relative translation for {!r} has near-zero length'.format(
                db_name))
        direction /= direction_norm
        if not args.dry_run:
            geometries.append(line_geometry(db_center,
                                            db_center + args.ray_length * direction,
                                            db_color))
        loaded_pairs += 1

    if loaded_pairs == 0:
        raise RuntimeError('No valid database poses found in {}'.format(details_path))

    print('Query:', query_name)
    print('Estimated query pose source:',
          'pose_details final c2w' if details_estimated_c2w is not None
          else str(query_poses_path))
    print('Loaded database poses:', loaded_pairs)
    print('Rendered geometries:', len(geometries) if not args.dry_run
          else 1 + 2 * loaded_pairs)
    if args.dry_run:
        return

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    viewer = o3d.visualization.O3DVisualizer(
        'Reloc3r InLoc localization: {}'.format(query_name), 1280, 800)
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
