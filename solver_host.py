from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PLUGIN_PARENT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

import wt_splatting  # noqa: E402,F401
from witwin_server.features.solvers.sdk import serve  # noqa: E402
from wt_splatting.reconstruction.config import ReconstructionConfig  # noqa: E402
from wt_splatting.reconstruction.pipeline import ReconstructionPipeline  # noqa: E402


def solve(ctx, scene, config):
    config = dict(config or {})
    project_root = _project_root(config, scene)
    reconstruction_config = ReconstructionConfig.from_component_values(_component_values(config))
    result = ReconstructionPipeline().run(
        reconstruction_config,
        project_root,
        lambda fraction, message: ctx.progress(fraction, message),
    )
    return {
        "run_id": result.run_id,
        "splat_path": str(result.splat_path),
        "splat_count": result.splat_count,
        "format": result.format,
        "summary": result.summary,
    }


def _project_root(config: dict[str, Any], scene) -> Path:
    value = config.get("project_root")
    if value:
        return Path(value).resolve()
    server = getattr(scene, "_server", None)
    for attr in ("project_root", "project_dir"):
        candidate = getattr(server, attr, None)
        if candidate:
            return Path(candidate).resolve()
    env_value = os.environ.get("WITWIN_PROJECT_DIR")
    if env_value:
        return Path(env_value).resolve()
    raise ValueError("project_root is required")


def _component_values(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for section in ("image", "colmap", "training", "export"):
        values.update(dict(config.get(section) or {}))
    for key, value in config.items():
        if key not in {"project_root", "image", "colmap", "training", "export"}:
            values[key] = value
    return values


if __name__ == "__main__":
    serve(solve)
