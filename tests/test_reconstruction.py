from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_config_normalizes_component_values() -> None:
    from wt_splatting.reconstruction.config import ReconstructionConfig

    config = ReconstructionConfig.from_component_values(
        {
            "image_dir": "assets/splatting/captures/demo/images",
            "capture_id": "desk",
            "output_name": "desk_recon",
            "downscale_factor": "2",
            "max_images": "5",
            "colmap_binary": "colmap",
            "camera_model": "OPENCV",
            "matcher": "exhaustive",
            "device": "cpu",
            "max_steps": "10",
            "sh_degree": "2",
            "export_format": "ply",
        }
    )

    assert config.image.image_dir == "assets/splatting/captures/demo/images"
    assert config.image.capture_id == "desk"
    assert config.image.downscale_factor == 2
    assert config.image.max_images == 5
    assert config.colmap.matcher == "exhaustive"
    assert config.training.device == "cpu"
    assert config.training.max_steps == 10
    assert config.training.sh_degree == 2
    assert config.export.output_name == "desk_recon"
    assert config.export.export_format == "ply"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("downscale_factor", 3),
        ("matcher", "vocab_tree"),
        ("export_format", "splat"),
        ("image_dir", str(Path("C:/outside/images"))),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object) -> None:
    from wt_splatting.reconstruction.config import ReconstructionConfig

    values = {"image_dir": "assets/images"}
    values[field] = value

    with pytest.raises(ValueError):
        ReconstructionConfig.from_component_values(values)


def test_image_dir_resolution_stays_inside_project(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.images import resolve_project_relative_dir

    project = tmp_path / "project"
    image_dir = project / "assets" / "splatting" / "captures" / "demo" / "images"
    image_dir.mkdir(parents=True)

    assert resolve_project_relative_dir(project, "assets/splatting/captures/demo/images") == image_dir.resolve()
    with pytest.raises(ValueError):
        resolve_project_relative_dir(project, "../outside")
    with pytest.raises(ValueError):
        resolve_project_relative_dir(project, tmp_path / "outside")


def test_discover_images_uses_supported_suffixes_and_natural_sort(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.images import discover_images

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ["frame_10.jpg", "frame_2.jpg", "frame_1.png", "notes.txt", "FRAME_3.WEBP"]:
        (image_dir / name).write_bytes(b"x")

    assert [path.name for path in discover_images(image_dir)] == [
        "frame_1.png",
        "frame_2.jpg",
        "FRAME_3.WEBP",
        "frame_10.jpg",
    ]
    assert [path.name for path in discover_images(image_dir, max_images=2)] == ["frame_1.png", "frame_2.jpg"]


def test_build_image_manifest_copies_inputs_and_reads_dimensions(tmp_path: Path) -> None:
    from PIL import Image

    from wt_splatting.reconstruction.config import ImageSequenceConfig
    from wt_splatting.reconstruction.images import build_image_manifest

    project = tmp_path / "project"
    image_dir = project / "assets" / "splatting" / "captures" / "demo" / "images"
    run_dir = project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-1"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(255, 0, 0)).save(image_dir / "frame_2.jpg")
    Image.new("RGB", (4, 3), color=(0, 255, 0)).save(image_dir / "frame_10.png")

    manifest = build_image_manifest(
        project,
        ImageSequenceConfig(
            image_dir="assets/splatting/captures/demo/images",
            capture_id="demo",
            downscale_factor=1,
            max_images=0,
        ),
        run_dir,
    )

    assert manifest.image_count == 2
    assert [entry.original_path.name for entry in manifest.images] == ["frame_2.jpg", "frame_10.png"]
    assert all(entry.staged_path.exists() for entry in manifest.images)
    assert manifest.images[0].width == 8
    assert manifest.images[0].height == 6
    assert manifest.warnings == []
    assert (run_dir / "input_manifest.json").exists()


class _Context:
    def __init__(self, name: str) -> None:
        self.name = name
        self.position = [0.0, 0.0, 0.0]
        self.parent_id = None
        self.scene = None
