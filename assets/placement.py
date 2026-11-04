"""Place a ``.wtsplat`` asset in the current Scene (the ``splat_asset_create_object`` scene op).

Studio sends this op when a ``.wtsplat`` file is dropped on the viewport or
placed from the Project panel (``createSplatAssetObject`` in Studio's
``features/assets/assetPlacement.ts``). ``scene.apply`` runs the op this
Plugin adds with studio.scene's ``provide_scene_op``, so the op log records
the new object like any other scene edit. (Until D16 P9 it was a raw message
handler, ``ctx.handlers.register``, which Studio removed.) The code lived in Studio's
kernel until 2026-11-03, where it had to import this Plugin (D16 review,
B-SPLAT).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from witwin_server.api import (
    BinaryAssetService,
    PluginContext,
    ProjectPathError,
    SceneObject,
    SceneWriteRejected,
    get_logger,
    resolve_project_path,
)

from witwin_server.plugins_builtin.scene.api import provide_scene_op

from .splat_manifest import read_wtsplat_manifest

logger = get_logger("SplatAssets")

# The scene op type; Studio's SCENE_OP_TYPES lists it, this Plugin answers it.
MESSAGE = "splat_asset_create_object"


def register(ctx: PluginContext) -> None:
    """Serve the ``splat_asset_create_object`` scene op; unloading the Plugin removes it."""
    async def op(server: Any, message: Dict[str, Any], _client: Any) -> Dict[str, Any]:
        return await create_splat_object(server, message)

    ctx.track(provide_scene_op(MESSAGE, op))


async def create_splat_object(server: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """Add one object with a ``GaussianSplatRenderer`` for the ``.wtsplat`` at ``data["path"]``.

    ``data``: ``path`` (Project path of the manifest), optional ``position``
    (world metres, default origin). Replies ``splat_asset_create_object_result``,
    ``scene_write_rejected`` during a timeline replay, or ``error``.
    """
    root = Path(server.default_scene_dir).resolve()
    rel_path = str(data.get("path") or "")
    target, error = _resolve(root, rel_path)
    if error or target is None or not target.is_file():
        return {"type": "error", "message": error or "Splat asset not found"}
    if target.suffix.lower() != ".wtsplat":
        return {"type": "error", "message": "Splat assets must use .wtsplat manifests"}

    scene = server.scene
    rejected = _require_author_write(server, scene)
    if rejected:
        return rejected

    try:
        manifest = await asyncio.to_thread(read_wtsplat_manifest, target, project_root=root)
        payload = await asyncio.to_thread(manifest.source_path.read_bytes)
    except Exception as exc:  # noqa: BLE001 - a bad file is the user's input, answer with the reason
        logger.warning("Failed to read splat asset %s: %s", rel_path, exc)
        return {"type": "error", "message": f"Failed to read splat asset: {exc}"}

    service = _asset_service(server)
    ref = service.register_bytes(
        kind="gaussian_splats",
        format=manifest.format,
        data=payload,
        scene_id=getattr(scene, "scene_id", "default"),
        shape=[manifest.splat_count],
        attributes={
            "splat_kind": "3dgs",
            "splat_count": manifest.splat_count,
            "source": manifest.source_rel,
            "manifest": rel_path.replace("\\", "/").lstrip("/"),
        },
        asset_id=manifest.source_path.name,
    )

    name = target.stem or manifest.source_path.stem or "Gaussian Splats"
    obj = SceneObject(name=name, mesh_type="Empty")
    renderer = obj.add_component("GaussianSplatRenderer")
    if renderer is None:
        return {"type": "error", "message": "GaussianSplatRenderer component is not registered"}
    scene.add_object(obj)
    renderer.set_splats_ref(ref.to_dict(), splat_count=manifest.splat_count, splat_kind="3dgs",
                            format=manifest.format)
    _set_position(obj, data.get("position") or [0.0, 0.0, 0.0])

    return {
        "type": "splat_asset_create_object_result",
        "object_id": obj.id,
        "path": rel_path,
        "source": manifest.source_rel,
        "splat_count": manifest.splat_count,
    }


def _resolve(root: Path, rel_path: str) -> Tuple[Optional[Path], Optional[str]]:
    try:
        return resolve_project_path(root, rel_path), None
    except ProjectPathError:
        logger.warning("Rejected splat asset path outside project: %r", rel_path)
        return None, "Path is outside the Project"


def _require_author_write(server: Any, scene: Any) -> Optional[Dict[str, Any]]:
    service = getattr(server, "scene_time", None)
    if service is None:
        return None
    try:
        service.require_author_write(scene.scene_id, operation=MESSAGE)
    except SceneWriteRejected as exc:
        return {
            "type": "scene_write_rejected",
            "scene_id": exc.scene_id,
            "operation": exc.operation,
            "mode": exc.mode.value,
            "message": str(exc),
        }
    return None


def _asset_service(server: Any) -> BinaryAssetService:
    handler = getattr(server, "handlers", {}).get("binary_assets")
    service = getattr(handler, "service", None)
    if not isinstance(service, BinaryAssetService):
        raise RuntimeError("BinaryAssetHandler not available")
    return service


def _set_position(obj: SceneObject, position: List[float]) -> None:
    transform = obj.get_component("Transform")
    if transform is None:
        return
    transform.position = torch.tensor(list(position[:3]), dtype=torch.float32)
