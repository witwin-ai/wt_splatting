from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PLUGIN_PARENT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

import wt_splatting  # noqa: E402,F401
from witwin.studio.solver import serve  # noqa: E402
from wt_splatting.meshization.config import MeshizationConfig  # noqa: E402
from wt_splatting.meshization.pipeline import MeshizationPipeline  # noqa: E402


def solve(ctx, scene, config):
    config = dict(config or {})
    project_root = _project_root(config, scene)
    mesh_config = MeshizationConfig.from_component_values(_component_values(config), project_root=project_root)
    ctx.progress(0.0, "start meshization")
    result = MeshizationPipeline().run(mesh_config, project_root, lambda fraction, message: ctx.progress(fraction, message))
    return {
        "run_id": result.run_id,
        "mesh_path": str(result.mesh_path),
        "vertex_count": result.vertex_count,
        "face_count": result.face_count,
        "format": result.format,
        "summary": result.summary,
    }


def _project_root(config: dict[str, Any], scene) -> Path:
    server = getattr(scene, "_server", None)
    for attr in ("project_root", "project_dir"):
        candidate = getattr(server, attr, None)
        if candidate:
            root = Path(candidate).resolve()
            _validate_config_root(config, root)
            return root
    env_value = os.environ.get("WITWIN_PROJECT_DIR")
    if env_value:
        root = Path(env_value).resolve()
        _validate_config_root(config, root)
        return root
    value = config.get("project_root")
    if value:
        return Path(value).resolve()
    raise ValueError("project_root is required")


def _validate_config_root(config: dict[str, Any], trusted_root: Path) -> None:
    value = config.get("project_root")
    if not value:
        return
    supplied = Path(value).resolve()
    if supplied != trusted_root:
        raise ValueError("project_root does not match the active Project")


def _component_values(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for section in ("source", "mesh"):
        values.update(dict(config.get(section) or {}))
    for key, value in config.items():
        if key not in {"project_root", "source", "mesh"}:
            values[key] = value
    return values


if __name__ == "__main__":
    serve(solve)
