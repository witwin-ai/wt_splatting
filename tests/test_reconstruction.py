from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "plugins"))


def test_reconstruction_component_defaults_and_solver_button() -> None:
    from wt_splatting.components.reconstruction import SplatReconstructionComponent

    component = SplatReconstructionComponent()
    payload = component.to_dict()
    values = payload["values"]

    assert values["image_dir"] == ""
    assert values["capture_id"] == "capture"
    assert values["output_name"] == "reconstruction"
    assert values["downscale_factor"] == 1
    assert values["max_images"] == 0
    assert values["colmap_binary"] == "colmap"
    assert values["camera_model"] == "OPENCV"
    assert values["matcher"] == "sequential"
    assert values["device"] == "cuda"
    assert values["max_steps"] == 7000
    assert values["sh_degree"] == 3
    assert values["export_format"] == "ply"
    assert values["status"] == "idle"
    assert values["last_run_id"] == ""
    assert values["last_error"] == ""

    buttons = {button["name"]: button for button in payload["buttons"]}
    assert buttons["reconstruct"]["display_name"] == "Reconstruct 3DGS"
    assert buttons["reconstruct"]["operation"] == "solver"
    assert buttons["reconstruct"]["cancellable"] is True
    assert buttons["reconstruct"]["progress_surface"] == "component"


def test_reconstruction_prefab_adds_reconstruction_and_renderer() -> None:
    from wt_splatting.library_items import _make_3dgs_reconstruction

    obj = _make_3dgs_reconstruction(_Context(name="3DGS Reconstruction"))

    assert obj.mesh_type == "Empty"
    assert obj.get_component("SplatReconstruction") is not None
    renderer = obj.get_component("GaussianSplatRenderer")
    assert renderer.to_dict()["values"]["splat_kind"] == "3dgs"
    assert renderer.to_dict()["values"]["splats"] is None


class _Context:
    def __init__(self, name: str) -> None:
        self.name = name
        self.position = [0.0, 0.0, 0.0]
        self.parent_id = None
        self.scene = None
