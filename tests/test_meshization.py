from __future__ import annotations

import struct
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "plugins"))


def test_mesh_config_normalizes_component_values(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    source = project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-1" / "model.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_gaussian_ply_bytes())

    config = MeshizationConfig.from_component_values(
        {
            "splat_path": str(source),
            "output_name": "kitchen_mesh",
            "grid_resolution": "40",
            "iso_level": "0.25",
            "opacity_threshold": "0.05",
            "max_gaussians": "1000",
            "sigma_radius": "2.5",
            "export_format": "ply",
        },
        project_root=project,
    )

    assert config.splat_path == source.resolve()
    assert config.output_name == "kitchen_mesh"
    assert config.grid_resolution == 40
    assert config.iso_level == pytest.approx(0.25)
    assert config.opacity_threshold == pytest.approx(0.05)
    assert config.max_gaussians == 1000
    assert config.sigma_radius == pytest.approx(2.5)
    assert config.export_format == "ply"


def test_mesh_config_rejects_sources_outside_project(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig

    outside = tmp_path / "outside.ply"
    outside.write_bytes(_gaussian_ply_bytes())

    with pytest.raises(ValueError, match="inside the Project"):
        MeshizationConfig.from_component_values({"splat_path": str(outside)}, project_root=tmp_path / "project")


def test_read_gaussian_ply_decodes_original_and_direct_values(tmp_path: Path) -> None:
    from wt_splatting.meshization.ply import read_gaussian_ply

    path = tmp_path / "model.ply"
    path.write_bytes(_gaussian_ply_bytes())

    splats = read_gaussian_ply(path)

    assert splats.count == 2
    assert splats.centers.shape == (2, 3)
    assert splats.scales.shape == (2, 3)
    assert splats.colors.shape == (2, 3)
    assert splats.opacities[0] == pytest.approx(0.8, abs=1e-5)
    assert np.all(splats.scales > 0)
    assert np.all((splats.colors >= 0.0) & (splats.colors <= 1.0))
    assert splats.colors[0, 0] == pytest.approx(0.8 * 0.28209479177387814 + 0.5)


def test_write_mesh_ply_rejects_mismatched_colors(tmp_path: Path) -> None:
    from wt_splatting.meshization.ply import write_mesh_ply

    vertices = np.zeros((2, 3), dtype=np.float32)
    faces = np.asarray([[0, 1, 1]], dtype=np.int32)
    colors = np.ones((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="colors"):
        write_mesh_ply(tmp_path / "mesh.ply", vertices, faces, colors)


def test_meshization_pipeline_writes_mesh_and_metadata(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig
    from wt_splatting.meshization.pipeline import MeshizationPipeline

    project = tmp_path / "project"
    source = project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-1" / "model.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_gaussian_ply_bytes())

    config = MeshizationConfig.from_component_values(
        {
            "splat_path": str(source),
            "output_name": "room_mesh",
            "grid_resolution": 12,
            "iso_level": 0.1,
            "export_format": "ply",
        },
        project_root=project,
    )

    pipeline = MeshizationPipeline(extractor=_fake_extractor)
    result = pipeline.run(config, project)

    assert result.mesh_path.name == "room_mesh.ply"
    assert result.mesh_path.exists()
    assert result.vertex_count == 4
    assert result.face_count == 2
    assert result.format == "mesh.ply"
    assert (result.run_dir / "meshization_result.json").exists()


def test_mesh_solver_host_returns_serializable_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wt_splatting.mesh_solver_host as host

    project = tmp_path / "project"
    source = project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-1" / "model.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_gaussian_ply_bytes())

    class FakePipeline:
        def run(self, config, project_root, progress=None):
            from wt_splatting.meshization.pipeline import MeshizationResult

            mesh_path = project_root / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "run-host" / "mesh.ply"
            mesh_path.parent.mkdir(parents=True)
            mesh_path.write_text("ply\n", encoding="ascii")
            return MeshizationResult(
                run_id="run-host",
                run_dir=mesh_path.parent,
                mesh_path=mesh_path,
                vertex_count=4,
                face_count=2,
                format="mesh.ply",
                summary="4 vertices, 2 faces",
            )

    monkeypatch.setattr(host, "MeshizationPipeline", lambda: FakePipeline())
    ctx = _Ctx()

    result = host.solve(
        ctx,
        scene=None,
        config={"project_root": str(project), "splat_path": str(source), "output_name": "mesh"},
    )

    assert result["run_id"] == "run-host"
    assert result["vertex_count"] == 4
    assert result["face_count"] == 2
    assert result["mesh_path"].endswith("mesh.ply")
    assert ctx.events


def test_mesh_solver_host_rejects_mismatched_config_project_root(tmp_path: Path) -> None:
    import wt_splatting.mesh_solver_host as host

    active = tmp_path / "active"
    supplied = tmp_path / "supplied"
    source = active / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-1" / "model.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_gaussian_ply_bytes())
    scene = type("Scene", (), {"_server": type("Server", (), {"project_root": active})()})()

    with pytest.raises(ValueError, match="active Project"):
        host.solve(
            _Ctx(),
            scene=scene,
            config={"project_root": str(supplied), "splat_path": str(source), "output_name": "mesh"},
        )


def test_mesh_solver_manifest_uses_supported_warm_lifecycle() -> None:
    manifest = tomllib.loads((ROOT / "plugins" / "wt_splatting" / "witwin.toml").read_text(encoding="utf-8"))
    solvers = {item["id"]: item for item in manifest["contributes"]["solvers"]}

    assert solvers["witwin.splatting.splat_to_mesh"]["lifecycle"] == "warm"
    assert solvers["witwin.splatting.reconstruct_3dgs"]["lifecycle"] == "warm"


def _fake_extractor(splats, config):
    vertices = np.asarray(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    colors = np.ones((4, 3), dtype=np.float32)
    return vertices, faces, colors


def _gaussian_ply_bytes() -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")
    rows = [
        (0.0, 0.0, 0.0, 0.8, 0.1, 0.2, 0.8, -2.3, -2.3, -2.3, 1.0, 0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0, 0.2, 0.7, 0.1, 0.6, -2.3, -2.3, -2.3, 1.0, 0.0, 0.0, 0.0),
    ]
    return header + b"".join(struct.pack("<14f", *row) for row in rows)


class _Ctx:
    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []

    def progress(self, fraction: float, message: str) -> None:
        self.events.append((fraction, message))
