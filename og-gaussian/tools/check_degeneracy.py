import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_RTOL = 3.0 * np.finfo(np.float32).eps


def check_alignment_matrix(matrix, *, rtol=DEFAULT_RTOL, atol=0.0):
    matrix = np.asarray(matrix)
    if matrix.shape != (3, 3) or matrix.dtype.kind not in 'fiu':
        raise ValueError('The alignment matrix must be a real numeric (3, 3) array.')
    matrix = matrix.astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('The alignment matrix contains NaN or infinity.')
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError('Rank tolerances must be finite and nonnegative.')

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if not np.isfinite(singular_values).all():
        raise ValueError('SVD produced non-finite singular values.')
    threshold = max(float(atol), float(rtol) * float(singular_values[0]))
    rank = int(np.count_nonzero(singular_values > threshold))
    return {
        'status': 'degenerate' if rank < 2 else 'pass',
        'rank': rank,
        'singular_values': singular_values.tolist(),
        'rank_threshold': threshold,
    }


def _xyz(array, name):
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] < 3 or array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be a real numeric (N, D) array with D >= 3.')
    xyz = array[:, :3].astype(np.float64)
    if len(xyz) == 0:
        raise ValueError(f'{name} is empty.')
    if not np.isfinite(xyz).all():
        raise ValueError(f'{name} coordinates contain NaN or infinity.')
    return xyz


def check_alignment(initial_xyz, predicted_xyz, *, rtol=DEFAULT_RTOL, atol=0.0):
    """Check paired centers before rigid alignment; only the first three columns count."""
    initial = _xyz(initial_xyz, 'Initial')
    predicted = _xyz(predicted_xyz, 'Predicted')
    if len(initial) != len(predicted):
        raise ValueError('Initial and predicted arrays must have the same number of points.')
    initial_centered = initial - initial.mean(axis=0, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=0, keepdims=True)
    alignment = initial_centered.T @ predicted_centered
    result = check_alignment_matrix(alignment, rtol=rtol, atol=atol)
    result['num_points'] = len(initial)
    return result


def _load_array(path):
    if path.suffix.lower() != '.npy':
        raise ValueError(f'Expected a numeric .npy array: {path}')
    return np.load(path, allow_pickle=False)


def _load_result(path, allow_pickle=False):
    keys = ('obj_gs_feat', 'goal_wo_svd')
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as result:
            return tuple(result[key] for key in keys)
    if path.suffix.lower() != '.npy':
        raise ValueError('Expected a .npz archive or a demo .npy result dictionary.')
    if not allow_pickle:
        raise ValueError('Demo .npy dictionaries require --allow-pickle for trusted local results.')
    result = np.load(path, allow_pickle=True)
    if not isinstance(result, np.ndarray) or result.shape != ():
        raise ValueError('Expected a scalar .npy result dictionary.')
    result = result.item()
    if not isinstance(result, dict):
        raise ValueError('Expected a .npy result dictionary.')
    return tuple(result[key] for key in keys)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Check whether the initial-to-terminal SVD alignment matrix has rank below 2.',
        epilog='Exit codes: 0 = pass; 1 = degenerate; 2 = invalid input. No input files are modified.',
    )
    parser.add_argument('result', nargs='?', type=Path, help='A demo .npy or .npz result archive.')
    parser.add_argument('--initial', type=Path, help='Initial coordinates or Gaussian attributes in .npy format.')
    parser.add_argument('--predicted', type=Path, help='Corresponding predicted terminal coordinates in .npy format.')
    parser.add_argument('--allow-pickle', action='store_true', help='Allow a trusted local demo .npy dictionary.')
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL, help='Relative singular-value tolerance.')
    parser.add_argument('--atol', type=float, default=0.0, help='Absolute singular-value tolerance.')
    args = parser.parse_args(argv)
    if args.result is not None:
        if args.initial is not None or args.predicted is not None:
            parser.error('Use either a result archive or --initial together with --predicted.')
    elif args.initial is None or args.predicted is None:
        parser.error('Provide a result archive or both --initial and --predicted.')

    try:
        if args.result is not None:
            initial, predicted = _load_result(args.result.expanduser(), args.allow_pickle)
        else:
            initial = _load_array(args.initial.expanduser())
            predicted = _load_array(args.predicted.expanduser())
        result = check_alignment(initial, predicted, rtol=args.rtol, atol=args.atol)
    except (OSError, ValueError, TypeError, KeyError, EOFError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result['status'] == 'degenerate')


if __name__ == '__main__':
    raise SystemExit(main())
