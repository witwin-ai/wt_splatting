from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PLUGIN_PARENT = Path(__file__).resolve().parents[3]
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

from wt_splatting.tools.adapters._common import copy_or_convert_mesh, find_newest_mesh, missing_modules, require_repo_file, run_logged


def build_sugar_command(
    *,
    repo: str | Path,
    scene_path: str | Path,
    iterations: int,
    python: str | Path,
    regularization: str = "sdf",
) -> list[str]:
    root = Path(repo).resolve()
    train_script = require_repo_file(root, "train_full_pipeline.py")
    return [
        str(Path(python).resolve()),
        str(train_script),
        "--scene_path",
        str(Path(scene_path).resolve()),
        "--regularization_type",
        regularization,
        "--refinement_iterations",
        str(iterations),
        "--high_poly",
        "True",
        "--export_obj",
        "True",
        "--export_ply",
        "True",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.repo:
        raise SystemExit("--repo is required for the official SuGaR adapter")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_modules(["open3d", "pytorch3d"])
    if missing:
        raise RuntimeError(f"Missing SuGaR dependencies in backend Python: {', '.join(missing)}")
    started = time.time() - 1.0
    command = build_sugar_command(
        repo=args.repo,
        scene_path=args.input,
        iterations=args.iterations,
        python=Path(sys.executable),
        regularization=args.regularization,
    )
    repo = Path(args.repo).resolve()
    run_logged(command, cwd=repo, log_path=output_dir / "sugar_official.log")
    mesh = _find_sugar_mesh(repo=repo, output_dir=output_dir, newer_than=started)
    copy_or_convert_mesh(mesh, args.mesh_output)
    return 0


def _find_sugar_mesh(*, repo: Path, output_dir: Path, newer_than: float) -> Path:
    for root in (output_dir, repo / "output", repo / "outputs"):
        if root.exists():
            try:
                return find_newest_mesh(root, newer_than=newer_than)
            except FileNotFoundError:
                pass
    log_tail = _tail(output_dir / "sugar_official.log")
    detail = f" SuGaR log tail: {log_tail}" if log_tail else ""
    raise FileNotFoundError(f"SuGaR did not produce a mesh under {output_dir} or {repo / 'output'}.{detail}")


def _tail(path: Path, *, max_chars: int = 2000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].replace("\n", " ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="wt_splatting adapter for the official SuGaR repository.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mesh-output", required=True)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--regularization", default="sdf", choices=["sdf", "density", "dn_consistency"])
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
