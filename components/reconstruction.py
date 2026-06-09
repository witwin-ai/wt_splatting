from __future__ import annotations

from typing import Any

from witwin_server.core.components.core import Component, button, component, int_field, string_field


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
        return "Reconstruction solver is not wired yet"
