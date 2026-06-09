from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parents[1]
for path in (REPO_ROOT / "server", REPO_ROOT / "plugins"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wt_splatting.meshization.config import MeshizationConfig  # noqa: E402
from wt_splatting.meshization.pipeline import MeshizationPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    events: list[tuple[float, str, float]] = []

    def progress(fraction: float, message: str) -> None:
        events.append((fraction, message, time.perf_counter()))
        print(f"{fraction:0.2f} {message}", flush=True)

    started = time.perf_counter()
    config = MeshizationConfig.from_component_values(
        {
            "splat_path": args.splat_path,
            "output_name": args.output_name,
            "grid_resolution": args.grid_resolution,
            "iso_level": args.iso_level,
            "opacity_threshold": args.opacity_threshold,
            "max_gaussians": args.max_gaussians,
            "sigma_radius": args.sigma_radius,
            "export_format": "ply",
        },
        project_root=args.project,
    )
    result = MeshizationPipeline().run(config, Path(args.project), progress)
    elapsed = time.perf_counter() - started
    print(f"run_id={result.run_id}")
    print(f"mesh_path={result.mesh_path}")
    print(f"vertex_count={result.vertex_count}")
    print(f"face_count={result.face_count}")
    print(f"elapsed_sec={elapsed:0.3f}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a wt_splatting 3DGS-to-mesh smoke test.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--splat-path", required=True)
    parser.add_argument("--output-name", default="splat_mesh")
    parser.add_argument("--grid-resolution", type=int, default=64)
    parser.add_argument("--iso-level", type=float, default=0.15)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--max-gaussians", type=int, default=50_000)
    parser.add_argument("--sigma-radius", type=float, default=3.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
