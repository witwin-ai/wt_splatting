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

from witwin_server.features.assets.binary.service import BinaryAssetService  # noqa: E402
from wt_splatting.io.loaders import register_3d_splats_runtime_file  # noqa: E402
from wt_splatting.reconstruction.config import ReconstructionConfig  # noqa: E402
from wt_splatting.reconstruction.pipeline import ReconstructionPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    events: list[tuple[float, str, float]] = []

    def progress(fraction: float, message: str) -> None:
        events.append((fraction, message, time.perf_counter()))
        print(f"{fraction:0.2f} {message}", flush=True)

    started = time.perf_counter()
    config = ReconstructionConfig.from_component_values(
        {
            "image_dir": args.image_dir,
            "capture_id": args.capture_id,
            "output_name": args.output_name,
            "downscale_factor": args.downscale_factor,
            "max_images": args.max_images,
            "colmap_binary": args.colmap_binary,
            "camera_model": args.camera_model,
            "matcher": args.matcher,
            "device": args.device,
            "max_steps": args.max_steps,
            "sh_degree": args.sh_degree,
            "export_format": "ply",
        }
    )
    result = ReconstructionPipeline().run(config, Path(args.project), progress)
    elapsed = time.perf_counter() - started

    print(f"run_id={result.run_id}")
    print(f"export_path={result.splat_path}")
    print(f"splat_count={result.splat_count}")
    print(f"elapsed_sec={elapsed:0.3f}")
    for index, (_, message, timestamp) in enumerate(events):
        previous = started if index == 0 else events[index - 1][2]
        print(f"phase {message}: {timestamp - previous:0.3f}s")

    if args.register:
        ref = register_3d_splats_runtime_file(
            BinaryAssetService(),
            result.splat_path,
            project_root=args.project,
            asset_id=args.asset_id or result.splat_path.name,
        )
        print(f"asset_id={ref.asset_id}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a wt_splatting reconstruction smoke test.")
    parser.add_argument("--project", required=True, help="Project root path.")
    parser.add_argument("--image-dir", required=True, help="Project-relative image sequence directory.")
    parser.add_argument("--capture-id", default="capture")
    parser.add_argument("--output-name", default="reconstruction")
    parser.add_argument("--downscale-factor", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--colmap-binary", default="colmap")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--register", action="store_true", help="Register the result in a local BinaryAssetService.")
    parser.add_argument("--asset-id", default="")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
