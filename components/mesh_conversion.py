from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from witwin_server import Notifications
from witwin_server.api import api
from witwin_server.core.components import MeshComponent
from witwin_server.core.components.core import Component, button, component, float_field, int_field, string_field

SOLVER_ID = "witwin.splatting.splat_to_mesh"


@component(name="SplatMeshConversion", category="Splatting")
class SplatMeshConversionComponent(Component):
    """User-facing controls for converting a 3DGS PLY into a triangle mesh."""

    splat_path = string_field("", group="Source")
    source_path = string_field("", group="Source")
    method = string_field("density", options=["density", "sugar", "2dgs"], enum_toggle=True, group="Source")
    output_name = string_field("splat_mesh", group="Mesh")
    grid_resolution = int_field(64, min=8, max=256, group="Mesh")
    iso_level = float_field(0.15, min=0.0, step=0.01, group="Mesh")
    opacity_threshold = float_field(0.01, min=0.0, max=1.0, step=0.01, group="Mesh")
    max_gaussians = int_field(50_000, min=1, group="Mesh")
    sigma_radius = float_field(3.0, min=0.25, max=8.0, step=0.25, group="Mesh")
    full_convergence_steps = int_field(30_000, min=1, group="Backend")
    backend_script = string_field("", group="Backend")
    backend_repo = string_field("", group="Backend")
    backend_python = string_field("", group="Backend")
    colmap_binary = string_field("colmap", group="2DGS COLMAP")
    camera_model = string_field("OPENCV", group="2DGS COLMAP")
    matcher = string_field("sequential", options=["sequential", "exhaustive"], enum_toggle=True, group="2DGS COLMAP")
    downscale_factor = int_field(1, min=1, group="2DGS COLMAP")
    max_images = int_field(0, min=0, group="2DGS COLMAP")
    mesh_resolution = int_field(1024, min=512, max=4096, group="Backend")
    mesh_cluster_count = int_field(1, min=1, max=1000, group="Backend")
    export_format = string_field("ply", options=["ply"], enum_toggle=True, group="Export")

    status = string_field("idle", readonly=True, group="Status")
    last_run_id = string_field("", readonly=True, group="Status")
    last_error = string_field("", readonly=True, group="Status")
    mesh_path = string_field("", readonly=True, group="Status")
    vertex_count = int_field(0, readonly=True, group="Status")
    face_count = int_field(0, readonly=True, group="Status")

    @button(
        display_name="Convert 3DGS to Mesh",
        group="Meshization",
        operation="solver",
        cancellable=True,
        progress_surface="component",
    )
    def convert_to_mesh(self, params: dict[str, Any] | None = None) -> str:
        async def _run():
            return await self._convert_async()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        return _run()

    async def _convert_async(self) -> str:
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
            return f"Meshization failed: {exc}"

        if getattr(run, "status", "") != "succeeded":
            message = (getattr(run, "error", None) or {}).get("message") or "meshization failed"
            self._set_status("failed", last_run_id=str(getattr(run, "run_id", "") or ""), last_error=message)
            Notifications.error("Splatting", message)
            return f"Meshization failed: {message}"

        outputs = _solver_payload(dict(getattr(run, "outputs", None) or {}))
        try:
            self._bind_solver_outputs(outputs)
        except Exception as exc:
            self._set_status("failed", last_run_id=str(getattr(run, "run_id", "") or ""), last_error=str(exc))
            Notifications.error("Splatting", str(exc))
            return f"Meshization failed: {exc}"

        run_id = str(outputs.get("run_id") or getattr(run, "run_id", "") or "")
        summary = str(outputs.get("summary") or "complete")
        self._set_status("complete", last_run_id=run_id, last_error="")
        Notifications.success("Splatting", "3DGS meshization complete")
        return f"Meshization complete: {summary}"

    def _solver_config(self) -> dict[str, Any]:
        return {
            "project_root": str(self._project_root()),
            "source": {
                "method": str(self.method),
                "splat_path": str(self.splat_path),
                "source_path": str(self.source_path or self.splat_path),
                "backend_script": str(self.backend_script),
                "backend_repo": str(self.backend_repo),
                "backend_python": str(self.backend_python),
                "colmap_binary": str(self.colmap_binary),
                "camera_model": str(self.camera_model),
                "matcher": str(self.matcher),
                "downscale_factor": int(self.downscale_factor),
                "max_images": int(self.max_images),
            },
            "mesh": {
                "output_name": str(self.output_name),
                "grid_resolution": int(self.grid_resolution),
                "iso_level": float(self.iso_level),
                "opacity_threshold": float(self.opacity_threshold),
                "max_gaussians": int(self.max_gaussians),
                "sigma_radius": float(self.sigma_radius),
                "full_convergence_steps": int(self.full_convergence_steps),
                "mesh_resolution": int(self.mesh_resolution),
                "mesh_cluster_count": int(self.mesh_cluster_count),
                "export_format": str(self.export_format),
            },
        }

    def _bind_solver_outputs(self, outputs: dict[str, Any]) -> None:
        mesh_path = str(outputs.get("mesh_path") or "")
        if not mesh_path:
            raise ValueError("Solver returned no mesh_path")
        self._set_status_fields(
            mesh_path=mesh_path,
            vertex_count=int(outputs.get("vertex_count") or 0),
            face_count=int(outputs.get("face_count") or 0),
        )
        self._bind_mesh_component(Path(mesh_path))

    def _bind_mesh_component(self, mesh_path: Path) -> None:
        owner = getattr(self, "_owner", None)
        if owner is None:
            return
        mesh_component = owner.get_component("Mesh") if hasattr(owner, "get_component") else None
        if mesh_component is None:
            mesh_component = MeshComponent(mesh_type="Custom")
            if hasattr(owner, "add_component"):
                owner.add_component(mesh_component)
            if hasattr(owner, "mesh_type"):
                owner.mesh_type = "Custom"
        loaded = trimesh.load(mesh_path, process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.uint32)
        mesh_component.set_mesh_data(vertices, faces, notify=True)

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

    def _set_status_fields(self, *, mesh_path: str, vertex_count: int, face_count: int) -> None:
        for field_name, value in {
            "mesh_path": mesh_path,
            "vertex_count": vertex_count,
            "face_count": face_count,
        }.items():
            self.set_field(field_name, value, silent=True, force=True)
            self._publish_component_field(field_name, value)

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


def _solver_payload(outputs: dict[str, Any]) -> dict[str, Any]:
    result = outputs.get("result")
    if isinstance(result, dict):
        return dict(result)
    return outputs
