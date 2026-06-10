from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PLUGIN_PARENT = Path(__file__).resolve().parents[3]
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

from wt_splatting.tools.adapters._common import copy_or_convert_mesh, find_newest_mesh, missing_modules, require_repo_file, run_logged


def build_2dgs_commands(
    *,
    repo: str | Path,
    scene_path: str | Path,
    model_dir: str | Path,
    iterations: int,
    python: str | Path,
    mesh_res: int = 1024,
    mesh_cluster_count: int = 1,
    unbounded: bool = True,
) -> tuple[list[str], list[str]]:
    root = Path(repo).resolve()
    train_script = require_repo_file(root, "train.py")
    render_script = require_repo_file(root, "render.py")
    scene = Path(scene_path).resolve()
    model = Path(model_dir).resolve()
    train_command = [
        str(Path(python).resolve()),
        str(train_script),
        "--source_path",
        str(scene),
        "--model_path",
        str(model),
        "--iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
    ]
    render_command = [
        str(Path(python).resolve()),
        str(render_script),
        "--source_path",
        str(scene),
        "--model_path",
        str(model),
        "--iteration",
        str(iterations),
        "--skip_train",
        "--skip_test",
        "--mesh_res",
        str(mesh_res),
        "--num_cluster",
        str(mesh_cluster_count),
    ]
    if unbounded:
        render_command.append("--unbounded")
    return train_command, render_command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.repo:
        raise SystemExit("--repo is required for the official 2DGS adapter")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_modules(["diff_surfel_rasterization", "simple_knn", "open3d"])
    if missing:
        raise RuntimeError(f"Missing 2DGS dependencies in backend Python: {', '.join(missing)}")
    model_dir = output_dir / "2dgs_model"
    train_command, render_command = build_2dgs_commands(
        repo=args.repo,
        scene_path=args.input,
        model_dir=model_dir,
        iterations=args.iterations,
        python=Path(sys.executable),
        mesh_res=args.mesh_res,
        mesh_cluster_count=args.mesh_cluster_count,
        unbounded=not args.bounded,
    )
    repo = Path(args.repo).resolve()
    started = time.time() - 1.0
    run_logged(train_command, cwd=repo, log_path=output_dir / "2dgs_train.log")
    run_logged(render_command, cwd=repo, log_path=output_dir / "2dgs_render.log")
    mesh = find_newest_mesh(model_dir if model_dir.exists() else output_dir, newer_than=started)
    copy_or_convert_mesh(mesh, args.mesh_output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="wt_splatting adapter for the official 2D Gaussian Splatting repository.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mesh-output", required=True)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mesh-res", type=int, default=1024)
    parser.add_argument("--mesh-cluster-count", type=int, default=1)
    parser.add_argument("--bounded", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
