from __future__ import annotations

import struct
import subprocess
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


def test_mesh_config_accepts_sugar_scene_directory(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    scene_dir = project / ".witwin" / "runtime" / "wt_splatting" / "benchmarks" / "mipnerf360" / "kitchen" / "input"
    scene_dir.mkdir(parents=True)
    backend_script = project / "scripts" / "run_sugar.py"
    backend_script.parent.mkdir(parents=True)
    backend_script.write_text("print('sugar')\n", encoding="utf-8")

    config = MeshizationConfig.from_component_values(
        {
            "method": "sugar",
            "source_path": str(scene_dir),
            "backend_script": str(backend_script),
            "backend_python": sys.executable,
            "full_convergence_steps": "30000",
        },
        project_root=project,
    )

    assert config.method == "sugar"
    assert config.source_path == scene_dir.resolve()
    assert config.backend_script == backend_script.resolve()
    assert config.backend_python == Path(sys.executable).resolve()
    assert config.full_convergence_steps == 30000


def test_mesh_config_rejects_unknown_method(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    source = project / "model.ply"
    project.mkdir()
    source.write_bytes(_gaussian_ply_bytes())

    with pytest.raises(ValueError, match="method"):
        MeshizationConfig.from_component_values({"splat_path": str(source), "method": "neus"}, project_root=project)


def test_mesh_config_rejects_sources_outside_project(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig

    outside = tmp_path / "outside.ply"
    outside.write_bytes(_gaussian_ply_bytes())

    with pytest.raises(ValueError, match="inside the Project"):
        MeshizationConfig.from_component_values({"splat_path": str(outside)}, project_root=tmp_path / "project")


def test_sugar_backend_builds_argument_list(tmp_path: Path) -> None:
    from wt_splatting.meshization.backends.sugar import SuGaRBackend
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    source = project / "scene"
    source.mkdir(parents=True)
    script = project / "scripts" / "sugar_adapter.py"
    script.parent.mkdir()
    script.write_text("print('adapter')\n", encoding="utf-8")
    config = MeshizationConfig.from_component_values(
        {"method": "sugar", "source_path": str(source), "backend_script": str(script), "backend_python": sys.executable},
        project_root=project,
    )

    command = SuGaRBackend().build_command(config, project / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "run")

    assert isinstance(command, list)
    assert command[0] == str(Path(sys.executable).resolve())
    assert str(script.resolve()) in command
    assert "--input" in command
    assert str(source.resolve()) in command


def test_2dgs_backend_reports_missing_script(tmp_path: Path) -> None:
    from wt_splatting.meshization.backends.twodgs import TwoDGSBackend
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    source = project / "scene"
    source.mkdir(parents=True)
    config = MeshizationConfig.from_component_values(
        {"method": "2dgs", "source_path": str(source), "backend_script": str(project / "missing.py")},
        project_root=project,
    )

    with pytest.raises(ValueError, match="backend_script"):
        TwoDGSBackend().build_command(config, project / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "run")


def test_2dgs_config_rejects_invalid_mesh_resolution(tmp_path: Path) -> None:
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    scene = project / "scene"
    scene.mkdir(parents=True)

    with pytest.raises(ValueError, match="mesh_resolution"):
        MeshizationConfig.from_component_values(
            {"method": "2dgs", "source_path": str(scene), "mesh_resolution": 256},
            project_root=project,
        )


def test_builtin_2dgs_backend_passes_mesh_resolution(tmp_path: Path) -> None:
    from wt_splatting.meshization.backends.twodgs import TwoDGSBackend
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    scene = project / "scene"
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "images").mkdir()
    config = MeshizationConfig.from_component_values(
        {
            "method": "2dgs",
            "source_path": str(scene),
            "backend_python": sys.executable,
            "mesh_resolution": 512,
            "mesh_cluster_count": 3,
        },
        project_root=project,
    )

    command = TwoDGSBackend().build_command(config, project / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "run")

    assert "--mesh-res" in command
    assert command[command.index("--mesh-res") + 1] == "512"
    assert "--mesh-cluster-count" in command
    assert command[command.index("--mesh-cluster-count") + 1] == "3"


def test_2dgs_backend_prepares_image_directory_with_colmap(tmp_path: Path) -> None:
    from wt_splatting.meshization.backends.base import BackendMeshResult
    from wt_splatting.meshization.backends.twodgs import TwoDGSBackend
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    images = project / "assets" / "captures" / "room"
    images.mkdir(parents=True)
    (images / "frame_1.jpg").write_bytes(b"jpg")
    (images / "frame_2.jpg").write_bytes(b"jpg")
    script = project / "scripts" / "adapter.py"
    script.parent.mkdir()
    script.write_text("print('adapter')\n", encoding="utf-8")
    config = MeshizationConfig.from_component_values(
        {
            "method": "2dgs",
            "source_path": str(images),
            "backend_script": str(script),
            "backend_python": sys.executable,
            "colmap_binary": "fake-colmap",
        },
        project_root=project,
    )

    class FakeColmapRunner:
        def __init__(self) -> None:
            self.calls = []

        def run(self, colmap_config, manifest, run_dir):
            from wt_splatting.reconstruction.colmap import ColmapRunResult

            self.calls.append((colmap_config, manifest, run_dir))
            dense = run_dir / "colmap" / "dense"
            (dense / "sparse").mkdir(parents=True)
            (dense / "images").mkdir()
            (dense / "images" / "000001.jpg").write_bytes(b"jpg")
            (dense / "sparse" / "images.bin").write_bytes(b"images")
            (dense / "sparse" / "cameras.bin").write_bytes(b"cameras")
            (dense / "sparse" / "points3D.bin").write_bytes(b"points")
            return ColmapRunResult(True, run_dir / "colmap" / "sparse", dense, "ok", registered_images=2)

    class CapturingBackend(TwoDGSBackend):
        def _run_external(self, prepared_config, run_dir, progress=None):
            mesh_path = run_dir / "mesh.ply"
            mesh_path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n", encoding="ascii")
            return BackendMeshResult(mesh_path=mesh_path, vertex_count=0, face_count=0, method="2dgs", summary=str(prepared_config.source_path))

    runner = FakeColmapRunner()
    result = CapturingBackend(colmap_runner=runner).run(config, project / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "run")

    assert len(runner.calls) == 1
    colmap_config, manifest, run_dir = runner.calls[0]
    assert colmap_config.colmap_binary == "fake-colmap"
    assert manifest.image_count == 2
    assert run_dir.name == "run"
    prepared = Path(result.summary)
    assert prepared.name == "2dgs_scene"
    assert (prepared / "sparse" / "0" / "images.bin").exists()
    assert (prepared / "images" / "000001.jpg").exists()


def test_2dgs_backend_uses_existing_colmap_scene_without_prepare(tmp_path: Path) -> None:
    from wt_splatting.meshization.backends.base import BackendMeshResult
    from wt_splatting.meshization.backends.twodgs import TwoDGSBackend
    from wt_splatting.meshization.config import MeshizationConfig

    project = tmp_path / "project"
    scene = project / "assets" / "captures" / "room_colmap"
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "images").mkdir()
    script = project / "scripts" / "adapter.py"
    script.parent.mkdir()
    script.write_text("print('adapter')\n", encoding="utf-8")
    config = MeshizationConfig.from_component_values(
        {"method": "2dgs", "source_path": str(scene), "backend_script": str(script), "backend_python": sys.executable},
        project_root=project,
    )

    class FailingColmapRunner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("COLMAP should not run for existing scene")

    class CapturingBackend(TwoDGSBackend):
        def _run_external(self, prepared_config, run_dir, progress=None):
            mesh_path = run_dir / "mesh.ply"
            mesh_path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n", encoding="ascii")
            return BackendMeshResult(mesh_path=mesh_path, vertex_count=0, face_count=0, method="2dgs", summary=str(prepared_config.source_path))

    result = CapturingBackend(colmap_runner=FailingColmapRunner()).run(
        config,
        project / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "run",
    )

    assert Path(result.summary) == scene.resolve()


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


def test_meshization_pipeline_dispatches_external_backend(tmp_path: Path) -> None:
    from wt_splatting.meshization.backends.base import BackendMeshResult
    from wt_splatting.meshization.config import MeshizationConfig
    from wt_splatting.meshization.pipeline import MeshizationPipeline

    project = tmp_path / "project"
    source = project / "scene"
    source.mkdir(parents=True)
    script = project / "scripts" / "adapter.py"
    script.parent.mkdir()
    script.write_text("print('adapter')\n", encoding="utf-8")
    config = MeshizationConfig.from_component_values(
        {"method": "sugar", "source_path": str(source), "backend_script": str(script), "output_name": "sugar_mesh"},
        project_root=project,
    )

    class FakeBackend:
        def run(self, cfg, run_dir, progress=None):
            mesh_path = run_dir / "sugar_mesh.ply"
            mesh_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nelement face 0\nend_header\n0 0 0\n", encoding="ascii")
            return BackendMeshResult(
                mesh_path=mesh_path,
                vertex_count=1,
                face_count=0,
                method="sugar",
                summary="fake sugar",
            )

    result = MeshizationPipeline(backends={"sugar": FakeBackend()}).run(config, project)

    assert result.mesh_path.name == "sugar_mesh.ply"
    assert result.vertex_count == 1
    assert result.face_count == 0
    assert result.format == "mesh.ply"


def test_mipnerf360_stages_local_scene_and_writes_report(tmp_path: Path) -> None:
    from wt_splatting.benchmarking.mipnerf360 import BenchmarkRecord, discover_scene_images, stage_scene_images, write_report

    dataset = tmp_path / "dataset"
    kitchen = dataset / "kitchen" / "images"
    room = dataset / "room" / "images"
    kitchen.mkdir(parents=True)
    room.mkdir(parents=True)
    (kitchen / "frame_2.jpg").write_bytes(b"jpg")
    (kitchen / "frame_10.jpg").write_bytes(b"jpg")
    (room / "frame_1.png").write_bytes(b"png")
    project = tmp_path / "project"

    images = discover_scene_images(dataset, "kitchen")
    staged = stage_scene_images(images, project, "kitchen", max_images=0)
    report = write_report(
        project / ".witwin" / "runtime" / "wt_splatting" / "benchmarks" / "mipnerf360",
        [
            BenchmarkRecord(
                scene="kitchen",
                method="sugar",
                status="succeeded",
                run_id="run-1",
                mesh_path=str(staged.scene_dir / "mesh.ply"),
                vertex_count=10,
                face_count=4,
                elapsed_sec=1.25,
                message="ok",
            )
        ],
    )

    assert [path.name for path in images] == ["frame_2.jpg", "frame_10.jpg"]
    assert (staged.scene_dir / "000001.jpg").exists()
    assert (staged.scene_dir / "000002.jpg").exists()
    staged_limited = stage_scene_images(images, project, "kitchen", max_images=1)
    staged_files = sorted(path.name for path in staged_limited.scene_dir.iterdir())
    assert staged_limited.image_count == 1
    assert staged_files == ["000001.jpg", "manifest.json"]
    assert report.exists()
    assert "kitchen" in report.read_text(encoding="utf-8")


def test_mipnerf360_benchmark_passes_colmap_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wt_splatting.benchmarking.mipnerf360 as bench

    dataset = tmp_path / "dataset"
    kitchen = dataset / "kitchen" / "images"
    kitchen.mkdir(parents=True)
    (kitchen / "frame_1.jpg").write_bytes(b"jpg")
    project = tmp_path / "project"
    seen = []

    class FakePipeline:
        def run(self, config, project_root):
            seen.append(config)
            from wt_splatting.meshization.pipeline import MeshizationResult

            mesh_path = project_root / ".witwin" / "runtime" / "wt_splatting" / "mesh" / "fake" / "mesh.ply"
            return MeshizationResult("fake", mesh_path.parent, mesh_path, 0, 0, "mesh.ply", "ok")

    monkeypatch.setattr(bench, "MeshizationPipeline", lambda: FakePipeline())

    records = bench.run_benchmark(
        project_root=project,
        dataset_root=dataset,
        scenes=["kitchen"],
        methods=["2dgs"],
        allow_download=False,
        colmap_binary="C:/tools/colmap.exe",
        camera_model="PINHOLE",
        matcher="exhaustive",
        downscale_factor=2,
    )

    assert records[0].status == "succeeded"
    assert seen[0].colmap_binary == "C:/tools/colmap.exe"
    assert seen[0].camera_model == "PINHOLE"
    assert seen[0].matcher == "exhaustive"
    assert seen[0].downscale_factor == 2


def test_official_sugar_adapter_builds_upstream_command(tmp_path: Path) -> None:
    from wt_splatting.tools.adapters.sugar_official import build_sugar_command

    repo = tmp_path / "SuGaR"
    repo.mkdir()
    (repo / "train_full_pipeline.py").write_text("print('train')\n", encoding="utf-8")
    scene = tmp_path / "scene"
    scene.mkdir()

    command = build_sugar_command(
        repo=repo,
        scene_path=scene,
        iterations=30000,
        python=Path(sys.executable),
        regularization="sdf",
    )

    assert command[:2] == [str(Path(sys.executable).resolve()), str(repo / "train_full_pipeline.py")]
    assert command[command.index("--scene_path") + 1] == str(scene.resolve())
    assert command[command.index("--regularization_type") + 1] == "sdf"
    assert command[command.index("--refinement_iterations") + 1] == "30000"
    assert "--high_poly" in command
    assert "--export_obj" in command


def test_official_2dgs_adapter_builds_train_and_render_commands(tmp_path: Path) -> None:
    from wt_splatting.tools.adapters.twodgs_official import build_2dgs_commands

    repo = tmp_path / "2d-gaussian-splatting"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')\n", encoding="utf-8")
    (repo / "render.py").write_text("print('render')\n", encoding="utf-8")
    scene = tmp_path / "room"
    scene.mkdir()
    output_dir = tmp_path / "out"

    train_command, render_command = build_2dgs_commands(
        repo=repo,
        scene_path=scene,
        model_dir=output_dir / "model",
        iterations=30000,
        python=Path(sys.executable),
        mesh_res=1024,
        mesh_cluster_count=1,
        unbounded=True,
    )

    assert train_command[:2] == [str(Path(sys.executable).resolve()), str(repo / "train.py")]
    assert train_command[train_command.index("--source_path") + 1] == str(scene.resolve())
    assert train_command[train_command.index("--model_path") + 1] == str((output_dir / "model").resolve())
    assert train_command[train_command.index("--iterations") + 1] == "30000"
    assert render_command[:2] == [str(Path(sys.executable).resolve()), str(repo / "render.py")]
    assert "--unbounded" in render_command
    assert render_command[render_command.index("--mesh_res") + 1] == "1024"
    assert render_command[render_command.index("--num_cluster") + 1] == "1"


def test_official_adapter_selects_newest_mesh(tmp_path: Path) -> None:
    from wt_splatting.tools.adapters._common import find_newest_mesh, missing_modules

    old_mesh = tmp_path / "old.ply"
    new_mesh = tmp_path / "nested" / "new.ply"
    new_mesh.parent.mkdir()
    old_mesh.write_text("ply\n", encoding="ascii")
    new_mesh.write_text("ply\n", encoding="ascii")
    old_time = 1_700_000_000
    new_time = 1_700_000_010
    old_mesh.touch()
    new_mesh.touch()
    import os

    os.utime(old_mesh, (old_time, old_time))
    os.utime(new_mesh, (new_time, new_time))

    assert find_newest_mesh(tmp_path) == new_mesh
    assert missing_modules(["sys", "definitely_missing_wt_splatting_backend_module"]) == ["definitely_missing_wt_splatting_backend_module"]


def test_official_adapter_run_logged_adds_backend_python_runtime_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from wt_splatting.tools.adapters import _common

    env_root = tmp_path / "env"
    python = env_root / "python.exe"
    python.parent.mkdir()
    python.write_text("", encoding="utf-8")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    _common.run_logged([str(python), "train.py"], cwd=tmp_path, log_path=tmp_path / "run.log")

    path_parts = seen["env"]["PATH"].split(_common.os.pathsep)
    assert path_parts[0] == str(env_root / "Library" / "bin")
    assert path_parts[1] == str(env_root)


def test_official_adapter_help_runs_from_foreign_cwd(tmp_path: Path) -> None:
    adapter = ROOT / "plugins" / "wt_splatting" / "tools" / "adapters" / "sugar_official.py"

    completed = subprocess.run(
        [sys.executable, str(adapter), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "official SuGaR" in completed.stdout


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
