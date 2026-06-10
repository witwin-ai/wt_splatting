from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parents[1]
for path in (REPO_ROOT / "server", REPO_ROOT / "plugins"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wt_splatting.benchmarking.mipnerf360 import DEFAULT_METHODS, DEFAULT_SCENES, run_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = run_benchmark(
        project_root=args.project,
        dataset_root=args.dataset_root,
        scenes=_split(args.scenes, DEFAULT_SCENES),
        methods=_split(args.methods, DEFAULT_METHODS),
        allow_download=not args.no_download,
        max_images=args.max_images,
        full_convergence_steps=args.full_convergence_steps if args.full_convergence else args.smoke_steps,
        backend_python=args.backend_python,
        sugar_script=args.sugar_script,
        twodgs_script=args.twodgs_script,
        sugar_repo=args.sugar_repo,
        twodgs_repo=args.twodgs_repo,
        colmap_binary=args.colmap_binary,
        camera_model=args.camera_model,
        matcher=args.matcher,
        downscale_factor=args.downscale_factor,
        mesh_resolution=args.mesh_resolution,
        mesh_cluster_count=args.mesh_cluster_count,
    )
    print(json.dumps([record.__dict__ for record in records], indent=2), flush=True)
    return 1 if any(record.status != "succeeded" for record in records) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run wt_splatting SuGaR/2DGS benchmarks on nvs-bench/mipnerf360.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--full-convergence", action="store_true")
    parser.add_argument("--full-convergence-steps", type=int, default=30_000)
    parser.add_argument("--smoke-steps", type=int, default=1000)
    parser.add_argument("--backend-python", default="")
    parser.add_argument("--sugar-script", default="")
    parser.add_argument("--twodgs-script", default="")
    parser.add_argument("--sugar-repo", default="")
    parser.add_argument("--twodgs-repo", default="")
    parser.add_argument("--colmap-binary", default="")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--mesh-resolution", type=int, default=1024)
    parser.add_argument("--mesh-cluster-count", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    return parser


def _split(value: str, default: tuple[str, ...]) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or list(default)


if __name__ == "__main__":
    raise SystemExit(main())
