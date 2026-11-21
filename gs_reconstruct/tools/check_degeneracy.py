import argparse
import csv
from pathlib import Path

import h5py
import torch


def check_alignment(initial_xyz, goal_xyz):
    initial = torch.as_tensor(initial_xyz, dtype=torch.float32, device="cpu")
    goal = torch.as_tensor(goal_xyz, dtype=torch.float32, device="cpu")
    if initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError("Initial coordinates must have shape (N, 3).")
    if initial.shape != goal.shape:
        raise ValueError("Initial and goal coordinates must have matching shapes.")
    if initial.shape[0] == 0:
        raise ValueError("The action point set is empty.")
    if not torch.isfinite(initial).all() or not torch.isfinite(goal).all():
        raise ValueError("Coordinates contain NaN or infinity.")

    initial_centered = initial - initial.mean(dim=0, keepdim=True)
    goal_centered = goal - goal.mean(dim=0, keepdim=True)
    alignment = initial_centered.T @ goal_centered
    if not torch.isfinite(alignment).all():
        raise ValueError("The alignment matrix contains NaN or infinity.")

    rank = int(torch.linalg.matrix_rank(alignment).item())
    singular_values = torch.linalg.svdvals(alignment)
    threshold = singular_values[0] * max(alignment.shape) * torch.finfo(alignment.dtype).eps
    return {
        "status": "degenerate" if rank < 2 else "pass",
        "num_points": initial.shape[0],
        "rank": rank,
        "singular_value_1": singular_values[0].item(),
        "singular_value_2": singular_values[1].item(),
        "singular_value_3": singular_values[2].item(),
        "rank_threshold": threshold.item(),
    }


def check_h5(path):
    points = []
    with h5py.File(path, "r") as data:
        for key in ("action/gs_feat", "action/gs_goal_feat"):
            dataset = data[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"{key} must be a dataset.")
            if dataset.ndim != 2 or dataset.shape[1] < 3:
                raise ValueError(f"{key} must have shape (N, D), with D >= 3.")
            if dataset.dtype.kind not in "fiu":
                raise ValueError(f"{key} must contain real numeric values.")
            points.append(dataset[:, :3])
    return check_alignment(*points)


def main():
    parser = argparse.ArgumentParser(
        description="Check action SVD alignment rank in generated H5 datasets.",
        epilog=(
            "Directories are searched recursively, including train/test folders. "
            "Exit codes: 0 = all pass; 1 = degenerate or invalid samples; "
            "2 = invalid arguments or report failure."
        ),
    )
    parser.add_argument("input", type=Path, help="An H5 file or a dataset directory.")
    parser.add_argument(
        "--report", type=Path, help="Optional CSV output path; must not already exist."
    )
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    if input_path.is_dir():
        files = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in (".h5", ".hdf5")
        )
    elif input_path.is_file() and input_path.suffix.lower() in (".h5", ".hdf5"):
        files = [input_path]
    else:
        parser.error(f"Expected an H5 file or directory: {input_path}")
    if not files:
        parser.error(f"No H5 files found in: {input_path}")

    report_path = args.report.expanduser() if args.report else None
    if report_path is not None and report_path.exists():
        parser.error(f"Report already exists: {report_path}")

    results = []
    counts = {"pass": 0, "degenerate": 0, "error": 0}
    for path in files:
        result = {"file": str(path), "error": ""}
        try:
            result.update(check_h5(path))
            values = ", ".join(
                f"{result[f'singular_value_{i}']:.6g}" for i in range(1, 4)
            )
            detail = f"rank={result['rank']}, singular_values=[{values}]"
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            result.update(status="error", error=str(exc))
            detail = str(exc)
        counts[result["status"]] += 1
        results.append(result)
        print(f"[{result['status'].upper()}] {path}: {detail}")

    print(
        f"Total: {len(files)}, pass: {counts['pass']}, "
        f"degenerate: {counts['degenerate']}, error: {counts['error']}"
    )
    if report_path is not None:
        fields = [
            "file", "status", "num_points", "rank", "singular_value_1",
            "singular_value_2", "singular_value_3", "rank_threshold", "error",
        ]
        try:
            with report_path.open("x", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
        except OSError as exc:
            parser.error(f"Cannot write report: {exc}")
        print(f"Report: {report_path}")
    return 1 if counts["degenerate"] or counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
