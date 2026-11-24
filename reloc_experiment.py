"""将 Reloc3r Aachen 推理结果整理为逐 query CSV。

示例：
    python reloc_experiment.py --dataset-dir D:\\data\\images \\
        --pairs D:\\hloc\\pairs.txt --model-dir D:\\hloc\\sfm \\
        --output-dir .\\output\\experiment --topk 5 --query-list queries.txt

query-list 为可选文本文件，每行一个 query 路径。CSV 会包含清单中的
所有 query；没有匹配 pairs 的 query 会标为 no_pairs。模型推理、位姿
筛选和运动平均均由 reloc_aachen.py 与 reloc.py 执行。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

CSV_FIELDS = [
    'query',
    'status',
    'failure_reason',
    'pair_candidate_count',
    'valid_candidate_count',
    'retained_candidate_count',
    'elapsed_seconds',
    'qw', 'qx', 'qy', 'qz',
    'tx', 'ty', 'tz',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the existing Reloc3r Aachen localizer and write per-query results.')
    parser.add_argument('--dataset-dir', required=True,
                        help='Image root used to resolve paths from the pairs file.')
    parser.add_argument('--pairs', required=True,
                        help='HLoc query-reference pairs, in retrieval rank order.')
    parser.add_argument('--model-dir', required=True,
                        help='COLMAP/PyCOLMAP reconstruction for reference poses.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for query_poses.txt, diagnostics, and the result CSV.')
    parser.add_argument('--topk', type=int, default=5,
                        help='Maximum references used per query (default: 5).')
    parser.add_argument('--query-list', default=None,
                        help='Optional text file with one query path per line; includes queries without pairs.')
    parser.add_argument('--img-reso', choices=['224', '512'], default='512')
    parser.add_argument('--image-resize', choices=['stretch', 'pad'], default='pad')
    parser.add_argument('--rotation-threshold-deg', type=float, default=15.0)
    parser.add_argument('--no-rotation-consensus', action='store_true',
                        help='Disable the existing rotation-consensus filter.')
    parser.add_argument('--translation-parallel-threshold-deg', type=float, default=5.0)
    parser.add_argument('--no-translation-consensus', action='store_true',
                        help='Disable the existing translation-ray filter.')
    parser.add_argument('--amp', type=int, choices=[0, 1], default=0,
                        help='Use automatic mixed precision (1) or disable it (0).')
    args = parser.parse_args()
    if args.topk < 1:
        parser.error('--topk must be positive')
    return args


def _normalize_name(name):
    return name.replace('\\', '/').strip()


def _read_query_list(path):
    if path is None:
        return None
    queries = []
    seen = set()
    with Path(path).open('r', encoding='utf-8') as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith('#'):
                continue
            value = value.split()[0]
            key = _normalize_name(value)
            if key not in seen:
                queries.append(value)
                seen.add(key)
    return queries


def _match_pair_query(query, pair_names):
    normalized = _normalize_name(query)
    exact = [name for name in pair_names if _normalize_name(name) == normalized]
    if exact:
        return exact[0]
    basename = Path(normalized).name
    matches = [name for name in pair_names
               if Path(_normalize_name(name)).name == basename]
    return matches[0] if len(matches) == 1 else None


def _read_pose_file(path):
    poses = {}
    if not path.is_file():
        return poses
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith('#') or len(fields) < 9:
                continue
            try:
                pose_values = [float(value) for value in fields[1:8]]
                retained_count = int(fields[8])
            except ValueError:
                continue
            poses[fields[0]] = (pose_values, retained_count)
    return poses


def _file_state(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _read_query_diagnostics(details_dir, query, detail_is_fresh, translation_is_fresh):
    safe_name = query.replace('\\', '_').replace('/', '_')
    detail_path = details_dir / (safe_name + '.txt')
    translation_path = details_dir / (safe_name + '.trans.txt')
    reason = None
    valid_count = 0
    rotation_retained = None
    translation_retained = None

    if detail_is_fresh and detail_path.is_file():
        in_rotation_rows = False
        with detail_path.open('r', encoding='utf-8') as stream:
            for line in stream:
                stripped = line.strip()
                if stripped.startswith('valid:'):
                    valid_count += stripped.split(':', 1)[1].strip().lower() == 'true'
                if stripped.startswith('Localization failure:'):
                    reason = stripped.split(':', 1)[1].strip()
                if stripped.startswith('Rotation consensus (candidate'):
                    in_rotation_rows = True
                    rotation_retained = 0
                    continue
                if in_rotation_rows:
                    if stripped.startswith('Relative poses'):
                        in_rotation_rows = False
                    elif '/' in stripped:
                        rotation_retained += stripped.rsplit('/', 1)[-1].strip().lower() == 'yes'

    if translation_is_fresh and translation_path.is_file():
        with translation_path.open('r', encoding='utf-8') as stream:
            for line in stream:
                stripped = line.strip()
                if stripped.startswith('Retained candidates:'):
                    names = stripped.split(':', 1)[1].strip()
                    translation_retained = 0 if not names or names == 'None' else len(names.split(', '))
                    break
    retained = translation_retained
    if retained is None:
        retained = rotation_retained
    return reason, valid_count, retained


def _build_aachen_argv(args, query_names):
    argv = [
        '--dataset_dir', args.dataset_dir,
        '--pairs', args.pairs,
        '--model_dir', args.model_dir,
        '--output_dir', args.output_dir,
        '--topk', str(args.topk),
        '--img_reso', args.img_reso,
        '--image_resize', args.image_resize,
        '--rotation_threshold_deg', str(args.rotation_threshold_deg),
        '--translation_parallel_threshold_deg',
        str(args.translation_parallel_threshold_deg),
        '--amp', str(args.amp),
        '--pose_details',
    ]
    if args.no_rotation_consensus:
        argv.append('--no_rotation_consensus')
    if args.no_translation_consensus:
        argv.append('--no_translation_consensus')
    if query_names:
        argv.extend(['--query_image', *query_names])
    return argv


def apply_documented_translation_guard(retained_mask, details):
    """Apply the method document's 30%/two-candidate safety floor.

    The shared Reloc3r filter already computes ray distances, valid scores,
    median/MAD, and its threshold. Reuse those results and replace only its
    final 50% cancellation floor for this experiment adapter.
    """
    scores = np.asarray(details["scores"], dtype=np.float64)
    mask = np.asarray(retained_mask, dtype=bool).copy()
    valid = np.isfinite(scores)
    if int(valid.sum()) >= 3:
        threshold = float(details["threshold"])
        mask[:] = True
        mask[valid] = scores[valid] <= threshold
        minimum_remaining = max(2, int(np.ceil(0.3 * len(mask))))
        if int(mask.sum()) < minimum_remaining:
            mask[:] = True
        details["outlier_indices"] = np.flatnonzero(~mask).tolist()
        details["minimum_retained_candidates"] = minimum_remaining
    else:
        mask[:] = True
        details["outlier_indices"] = []
        details["minimum_retained_candidates"] = max(2, int(np.ceil(0.3 * len(mask))))
    return mask, details


def main():
    args = parse_args()
    import reloc_aachen

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = reloc_aachen.read_pairs(args.pairs)
    pair_names = list(pairs)
    query_list = _read_query_list(args.query_list)
    if query_list is None:
        entries = [(query, query) for query in pair_names]
    else:
        entries = [(query, _match_pair_query(query, pair_names)) for query in query_list]

    selected_pair_names = []
    for _, pair_query in entries:
        if pair_query is not None and pair_query not in selected_pair_names:
            selected_pair_names.append(pair_query)

    elapsed_by_query = {}
    run_error = None
    pose_file = output_dir / 'query_poses.txt'
    details_dir = output_dir / 'pose_details'
    previous_pose_state = _file_state(pose_file)
    previous_detail_states = {}
    # 记录运行前的文件状态，避免把上一次运行留下的位姿或诊断误认为本次结果。
    for _, pair_query in entries:
        if pair_query is not None:
            safe_name = pair_query.replace('\\', '_').replace('/', '_')
            for suffix in ('.txt', '.trans.txt'):
                path = details_dir / (safe_name + suffix)
                previous_detail_states[path] = _file_state(path)

    if selected_pair_names:
        original_argv = sys.argv[:]
        original_tqdm = reloc_aachen.tqdm
        original_translation_consensus = reloc_aachen.translation_ray_consensus

        # 包装原有 query 迭代器，在每个 query 的完整处理周期外侧计时，不介入定位算法。
        def timed_tqdm(iterable, *tqdm_args, **tqdm_kwargs):
            progress = original_tqdm(iterable, *tqdm_args, **tqdm_kwargs)
            if tqdm_kwargs.get('desc') != 'Localizing Aachen':
                return progress

            def timed_items():
                for item in progress:
                    query = item[0]
                    started = time.perf_counter()
                    try:
                        yield item
                    finally:
                        elapsed_by_query[query] = time.perf_counter() - started

            return timed_items()

        def documented_translation_consensus(*consensus_args, **consensus_kwargs):
            retained, details = original_translation_consensus(
                *consensus_args, **consensus_kwargs
            )
            return apply_documented_translation_guard(retained, details)

        reloc_aachen.tqdm = timed_tqdm
        reloc_aachen.translation_ray_consensus = documented_translation_consensus
        sys.argv = [str(Path(reloc_aachen.__file__).resolve()), *_build_aachen_argv(
            args, selected_pair_names)]
        try:
            reloc_aachen.main()
        except RuntimeError as exc:
            if str(exc) != 'No Aachen queries could be localized.':
                run_error = exc
        except Exception as exc:  # Keep a per-query report if inference aborts partway through.
            run_error = exc
        finally:
            reloc_aachen.tqdm = original_tqdm
            reloc_aachen.translation_ray_consensus = original_translation_consensus
            sys.argv = original_argv

    poses = _read_pose_file(pose_file)
    if selected_pair_names and _file_state(pose_file) == previous_pose_state:
        poses = {}
    rows = []
    for query, pair_query in entries:
        if pair_query is None:
            rows.append({
                'query': query,
                'status': 'no_pairs',
                'failure_reason': 'No matching query entry in the pairs file.',
                'pair_candidate_count': 0,
                'valid_candidate_count': 0,
                'retained_candidate_count': 0,
                'elapsed_seconds': '',
            })
            continue

        candidates = pairs[pair_query][:args.topk]
        safe_name = pair_query.replace('\\', '_').replace('/', '_')
        detail_path = details_dir / (safe_name + '.txt')
        translation_path = details_dir / (safe_name + '.trans.txt')
        reason, valid_count, diagnostic_retained = _read_query_diagnostics(
            details_dir, pair_query,
            _file_state(detail_path) != previous_detail_states.get(detail_path),
            _file_state(translation_path) != previous_detail_states.get(translation_path))
        pose_record = poses.get(pair_query)
        elapsed = elapsed_by_query.get(pair_query)
        row = {
            'query': query,
            'status': 'success' if pose_record is not None else 'failed',
            'failure_reason': '',
            'pair_candidate_count': len(candidates),
            'valid_candidate_count': valid_count,
            'retained_candidate_count': 0 if diagnostic_retained is None else diagnostic_retained,
            'elapsed_seconds': '' if elapsed is None else '{:.6f}'.format(elapsed),
        }
        if pose_record is not None:
            pose_values, retained_count = pose_record
            row['retained_candidate_count'] = retained_count
            row.update(dict(zip(CSV_FIELDS[7:14], pose_values)))
        else:
            if reason:
                row['failure_reason'] = reason
            elif run_error is not None:
                row['failure_reason'] = 'Reloc3r run aborted: {}'.format(run_error)
            elif elapsed is not None:
                row['failure_reason'] = 'Query image could not be loaded; check dataset-dir and query path.'
            else:
                row['failure_reason'] = 'Query was not processed by the Aachen localizer.'
        rows.append(row)

    csv_path = output_dir / 'reloc3r_results.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(row['status'] == 'success' for row in rows)
    print('Wrote {} query rows ({} localized) to {}'.format(
        len(rows), success_count, csv_path))
    if run_error is not None:
        raise RuntimeError('Reloc3r aborted after writing the per-query report: {}'.format(run_error))
    (output_dir / 'reloc3r_complete.json').write_text(
        json.dumps({
            'stage': 'reloc3r',
            'query_count': len(rows),
            'localized_count': success_count,
            'pairs': str(Path(args.pairs).resolve()),
            'topk': args.topk,
        }, indent=2) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
