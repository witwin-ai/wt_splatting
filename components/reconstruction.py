from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from witwin_server import Notifications
from witwin_server.api import api
from witwin_server.core.components.core import Component, button, component, int_field, string_field
from witwin_server.features.assets.binary.service import BinaryAssetService

from ..io.loaders import register_3d_splats_runtime_file

SOLVER_ID = "witwin.splatting.reconstruct_3dgs"

@component(name="SplatReconstruction", category="Splatting")
class SplatReconstructionComponent(Component):
    """User-facing controls for image-sequence-to-3DGS reconstruction."""

    image_dir = string_field("", group="Capture")
    capture_id = string_field("capture", group="Capture")
    output_name = string_field("reconstruction", group="Capture")
    downscale_factor = int_field(1, options=[1, 2, 4, 8], group="Capture")
    max_images = int_field(0, min=0, group="Capture")

    colmap_binary = string_field("colmap", group="COLMAP")
    camera_model = string_field("OPENCV", group="COLMAP")
    matcher = string_field("sequential", options=["sequential", "exhaustive"], enum_toggle=True, group="COLMAP")

    device = string_field("cuda", options=["cuda", "cpu"], enum_toggle=True, group="Training")
    max_steps = int_field(7000, min=1, group="Training")
    sh_degree = int_field(3, min=0, max=3, group="Training")
    export_format = string_field("ply", options=["ply"], enum_toggle=True, group="Export")

    status = string_field("idle", readonly=True, group="Status")
    last_run_id = string_field("", readonly=True, group="Status")
    last_error = string_field("", readonly=True, group="Status")

    @button(
        display_name="Reconstruct 3DGS",
        group="Reconstruction",
        operation="solver",
        cancellable=True,
        progress_surface="component",
    )
    def reconstruct(self, params: dict[str, Any] | None = None) -> str:
        async def _run():
            return await self._reconstruct_async()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        return _run()

    async def _reconstruct_async(self) -> str:
        self._set_status("running", last_error="")
        config = self._solver_config()
        try:
            run = await asyncio.to_thread(
                api.solvers.solve,
                SOLVER_ID,
                scene=getattr(self._owner, "scene", None),
                config=config,
            )
        except Exception as exc:
            self._set_status("failed", last_error=str(exc))
            Notifications.error("Splatting", str(exc))
            return f"Reconstruction failed: {exc}"

        if getattr(run, "status", "") != "succeeded":
            message = (getattr(run, "error", None) or {}).get("message") or "reconstruction failed"
            self._set_status("failed", last_run_id=str(getattr(run, "run_id", "") or ""), last_error=message)
            Notifications.error("Splatting", message)
            return f"Reconstruction failed: {message}"

        outputs = dict(getattr(run, "outputs", None) or {})
        try:
            self._bind_solver_outputs(outputs, config)
        except Exception as exc:
            self._set_status("failed", last_run_id=str(getattr(run, "run_id", "") or outputs.get("run_id") or ""), last_error=str(exc))
            Notifications.error("Splatting", str(exc))
            return f"Reconstruction failed: {exc}"

        run_id = str(outputs.get("run_id") or getattr(run, "run_id", "") or "")
        summary = str(outputs.get("summary") or "complete")
        self._set_status("complete", last_run_id=run_id, last_error="")
        Notifications.success("Splatting", "3DGS reconstruction complete")
        return f"Reconstruction complete: {summary}"

    def _solver_config(self) -> dict[str, Any]:
        return {
            "project_root": str(self._project_root()),
            "image": {
                "image_dir": str(self.image_dir),
                "capture_id": str(self.capture_id),
                "downscale_factor": int(self.downscale_factor),
                "max_images": int(self.max_images),
            },
            "colmap": {
                "colmap_binary": str(self.colmap_binary),
                "camera_model": str(self.camera_model),
                "matcher": str(self.matcher),
            },
            "training": {
                "device": str(self.device),
                "max_steps": int(self.max_steps),
                "sh_degree": int(self.sh_degree),
            },
            "export": {
                "output_name": str(self.output_name),
                "export_format": str(self.export_format),
            },
        }

    def _bind_solver_outputs(self, outputs: dict[str, Any], config: dict[str, Any]) -> None:
        splat_path = outputs.get("splat_path")
        if not splat_path:
            raise ValueError("Solver returned no splat_path")
        ref = register_3d_splats_runtime_file(
            self._asset_service(),
            splat_path,
            project_root=config["project_root"],
            asset_id=f"{self.output_name}-{outputs.get('run_id') or 'reconstruction'}.ply",
        )
        renderer = self._renderer_component()
        if renderer is None or not hasattr(renderer, "set_splats_ref"):
            raise ValueError("GaussianSplatRenderer sibling component is required")
        renderer.set_splats_ref(
            ref.to_dict(),
            splat_count=int(outputs.get("splat_count") or ref.shape[0] if ref.shape else 0),
            splat_kind="3dgs",
            format=str(outputs.get("format") or ref.format),
        )

    def _renderer_component(self):
        owner = getattr(self, "_owner", None)
        if owner is not None and hasattr(owner, "get_component"):
            return owner.get_component("GaussianSplatRenderer")
        return None

    def _asset_service(self) -> BinaryAssetService:
        scene = getattr(getattr(self, "_owner", None), "scene", None)
        server = getattr(scene, "_server", None)
        handler = getattr(server, "handlers", {}).get("binary_assets") if server is not None else None
        service = getattr(handler, "service", None)
        if not isinstance(service, BinaryAssetService):
            raise RuntimeError("BinaryAssetHandler not available")
        return service

    def _project_root(self) -> Path:
        scene = getattr(getattr(self, "_owner", None), "scene", None)
        for source in (scene, getattr(scene, "_server", None)):
            for attr in ("project_root", "project_dir"):
                value = getattr(source, attr, None)
                if value:
                    return Path(value).resolve()
        env_value = os.environ.get("WITWIN_PROJECT_DIR")
        if env_value:
            return Path(env_value).resolve()
        raise RuntimeError("Project root is not available")

    def _set_status(self, status: str, *, last_run_id: str | None = None, last_error: str | None = None) -> None:
        self.set_field("status", status, silent=True, force=True)
        self._publish_component_field("status", status)
        if last_run_id is not None:
            self.set_field("last_run_id", last_run_id, silent=True, force=True)
            self._publish_component_field("last_run_id", last_run_id)
        if last_error is not None:
            self.set_field("last_error", last_error, silent=True, force=True)
            self._publish_component_field("last_error", last_error)

    def _publish_component_field(self, field_name: str, value: Any) -> None:
        owner = getattr(self, "_owner", None)
        scene = getattr(owner, "scene", None)
        if scene is not None and hasattr(scene, "_emit"):
            scene._emit(
                "component_updated",
                object_id=getattr(owner, "id", ""),
                component_name=self._component_name,
                field_name=field_name,
                value=value,
            )
