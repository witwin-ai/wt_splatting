from __future__ import annotations

import sys
import tomllib
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "plugins"))

from witwin_server.features.assets.binary.service import BinaryAssetService


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


def test_colmap_commands_are_argument_lists(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import build_colmap_commands
    from wt_splatting.reconstruction.config import ColmapConfig
    from wt_splatting.reconstruction.images import ImageManifest

    manifest = ImageManifest(tmp_path / "images", tmp_path / "staged", 3, [], [])
    commands = build_colmap_commands(
        ColmapConfig(colmap_binary="colmap", camera_model="OPENCV", matcher="sequential"),
        manifest,
        tmp_path / "colmap",
    )

    assert [command[1] for command in commands] == [
        "feature_extractor",
        "sequential_matcher",
        "mapper",
        "image_undistorter",
    ]
    assert all(isinstance(command, list) for command in commands)
    assert all(all(isinstance(arg, str) for arg in command) for command in commands)


def test_colmap_runner_maps_missing_executable(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import ColmapRunner
    from wt_splatting.reconstruction.config import ColmapConfig
    from wt_splatting.reconstruction.images import ImageManifest

    def missing_runner(*args, **kwargs):
        raise FileNotFoundError("colmap")

    manifest = ImageManifest(tmp_path / "images", tmp_path / "staged", 3, [], [])
    result = ColmapRunner(run_process=missing_runner).run(
        ColmapConfig(colmap_binary="missing-colmap", matcher="exhaustive"),
        manifest,
        tmp_path / "run",
    )

    assert result.success is False
    assert "missing executable" in result.message


def test_colmap_runner_detects_sparse_model(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import ColmapRunner
    from wt_splatting.reconstruction.config import ColmapConfig
    from wt_splatting.reconstruction.images import ImageManifest

    calls = []

    def ok_runner(args, **kwargs):
        calls.append(args)
        if args[1] == "image_undistorter":
            (run_dir / "colmap" / "dense" / "sparse").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run_dir = tmp_path / "run"
    sparse_model = run_dir / "colmap" / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    (sparse_model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_1.jpg\n\n2 1 0 0 0 0 0 0 1 frame_2.jpg\n\n",
        encoding="utf-8",
    )
    manifest = ImageManifest(tmp_path / "images", tmp_path / "staged", 3, [], [])

    result = ColmapRunner(run_process=ok_runner).run(ColmapConfig(), manifest, run_dir)

    assert result.success is True
    assert result.sparse_dir == run_dir / "colmap" / "sparse"
    assert result.dense_dir == run_dir / "colmap" / "dense"
    assert len(calls) == 4


def test_colmap_runner_rejects_under_registered_text_model(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import ColmapRunner
    from wt_splatting.reconstruction.config import ColmapConfig
    from wt_splatting.reconstruction.images import ImageManifest

    def ok_runner(args, **kwargs):
        if args[1] == "image_undistorter":
            (run_dir / "colmap" / "dense" / "sparse").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run_dir = tmp_path / "run"
    sparse_model = run_dir / "colmap" / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    (sparse_model / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_1.jpg\n\n", encoding="utf-8")
    manifest = ImageManifest(tmp_path / "images", tmp_path / "staged", 3, [], [])

    result = ColmapRunner(run_process=ok_runner).run(ColmapConfig(), manifest, run_dir)

    assert result.success is False
    assert "too few registered images" in result.message


def test_colmap_runner_rejects_missing_dense_output(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import ColmapRunner
    from wt_splatting.reconstruction.config import ColmapConfig
    from wt_splatting.reconstruction.images import ImageManifest

    def ok_runner(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run_dir = tmp_path / "run"
    sparse_model = run_dir / "colmap" / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    (sparse_model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_1.jpg\n\n2 1 0 0 0 0 0 0 1 frame_2.jpg\n\n",
        encoding="utf-8",
    )
    manifest = ImageManifest(tmp_path / "images", tmp_path / "staged", 3, [], [])

    result = ColmapRunner(run_process=ok_runner).run(ColmapConfig(), manifest, run_dir)

    assert result.success is False
    assert "missing dense output" in result.message


def test_colmap_runner_counts_binary_sparse_model_without_pycolmap(monkeypatch, tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import ColmapRunner
    from wt_splatting.reconstruction.config import ColmapConfig
    from wt_splatting.reconstruction.images import ImageManifest

    monkeypatch.setitem(sys.modules, "pycolmap", None)

    def ok_runner(args, **kwargs):
        if args[1] == "image_undistorter":
            (run_dir / "colmap" / "dense" / "sparse").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run_dir = tmp_path / "run"
    sparse_model = run_dir / "colmap" / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    (sparse_model / "images.bin").write_bytes(struct.pack("<Q", 3))
    manifest = ImageManifest(tmp_path / "images", tmp_path / "staged", 3, [], [])

    result = ColmapRunner(run_process=ok_runner).run(ColmapConfig(), manifest, run_dir)

    assert result.success is True
    assert result.registered_images == 3


def test_gsplat_backend_reports_missing_dependencies(monkeypatch) -> None:
    from wt_splatting.reconstruction.gsplat_backend import GsplatBackend

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    availability = GsplatBackend().check_available()

    assert availability.available is False
    assert "torch" in availability.missing
    assert "gsplat" in availability.missing


def test_gsplat_training_config_serializes_reproducibly() -> None:
    from wt_splatting.reconstruction.config import GsplatTrainingConfig
    from wt_splatting.reconstruction.gsplat_backend import training_payload

    payload = training_payload(GsplatTrainingConfig(device="cpu", max_steps=12, sh_degree=2))

    assert payload == {"device": "cpu", "max_steps": 12, "sh_degree": 2}


def test_gsplat_backend_requires_adapter_before_training(monkeypatch) -> None:
    from wt_splatting.reconstruction.gsplat_backend import GsplatBackend

    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace())
    monkeypatch.delenv("WT_SPLATTING_GSPLAT_ADAPTER", raising=False)
    monkeypatch.delenv("WT_SPLATTING_GSPLAT_TRAIN_SCRIPT", raising=False)

    availability = GsplatBackend().check_available()

    assert availability.available is False
    assert "adapter" in availability.message


def test_fake_gsplat_backend_receives_colmap_dense_dir(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.colmap import ColmapRunResult
    from wt_splatting.reconstruction.config import GsplatTrainingConfig
    from wt_splatting.reconstruction.gsplat_backend import GsplatTrainResult

    class FakeBackend:
        def __init__(self) -> None:
            self.received_dense_dir = None

        def train(self, config, colmap_result, run_dir, progress):
            self.received_dense_dir = colmap_result.dense_dir
            export = run_dir / "gsplat" / "exports" / "model.ply"
            export.parent.mkdir(parents=True)
            export.write_bytes(_tiny_ply())
            return GsplatTrainResult(success=True, export_path=export, message="ok")

    backend = FakeBackend()
    colmap_result = ColmapRunResult(
        success=True,
        sparse_dir=tmp_path / "colmap" / "sparse",
        dense_dir=tmp_path / "colmap" / "dense",
        message="ok",
    )
    train_result = backend.train(GsplatTrainingConfig(device="cpu"), colmap_result, tmp_path, lambda *_: None)

    assert backend.received_dense_dir == colmap_result.dense_dir
    assert train_result.export_path == tmp_path / "gsplat" / "exports" / "model.ply"


def test_pipeline_runs_fake_backends_and_writes_result(tmp_path: Path) -> None:
    from PIL import Image

    from wt_splatting.reconstruction.config import ReconstructionConfig
    from wt_splatting.reconstruction.pipeline import ReconstructionPipeline

    project, image_rel = _make_project_images(tmp_path, count=2)
    progress = []
    pipeline = ReconstructionPipeline(
        colmap_runner=_FakeColmapRunner(),
        gsplat_backend=_FakeGsplatBackend(),
        run_id_factory=lambda: "run-fixed",
    )

    result = pipeline.run(
        ReconstructionConfig.from_component_values({"image_dir": image_rel, "device": "cpu", "max_steps": 4}),
        project,
        lambda fraction, message: progress.append((fraction, message)),
    )

    assert result.run_id == "run-fixed"
    assert result.splat_path.name == "reconstruction.ply"
    assert result.splat_count == 1
    assert [message for _, message in progress] == [
        "validate images",
        "build COLMAP model",
        "train gsplat model",
        "export splats",
        "write result",
    ]
    assert (project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-fixed" / "pipeline_result.json").exists()


def test_pipeline_uses_output_name_for_final_ply(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.config import ReconstructionConfig
    from wt_splatting.reconstruction.pipeline import ReconstructionPipeline

    project, image_rel = _make_project_images(tmp_path, count=2)
    pipeline = ReconstructionPipeline(
        colmap_runner=_FakeColmapRunner(),
        gsplat_backend=_FakeGsplatBackend(),
        run_id_factory=lambda: "run-named",
    )

    result = pipeline.run(
        ReconstructionConfig.from_component_values({"image_dir": image_rel, "output_name": "desk_scan"}),
        project,
    )

    assert result.splat_path.name == "desk_scan.ply"
    assert result.splat_path.exists()


def test_pipeline_failure_preserves_run_id_and_phase(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.config import ReconstructionConfig
    from wt_splatting.reconstruction.pipeline import ReconstructionPipeline, ReconstructionPipelineError

    project, image_rel = _make_project_images(tmp_path, count=2)
    pipeline = ReconstructionPipeline(
        colmap_runner=_FailingColmapRunner(),
        gsplat_backend=_FakeGsplatBackend(),
        run_id_factory=lambda: "run-failed",
    )

    with pytest.raises(ReconstructionPipelineError) as exc:
        pipeline.run(ReconstructionConfig.from_component_values({"image_dir": image_rel}), project)

    assert exc.value.run_id == "run-failed"
    assert exc.value.phase == "run_colmap"
    assert "run_colmap" in str(exc.value)


def test_pipeline_reports_export_phase_failures(tmp_path: Path) -> None:
    from wt_splatting.reconstruction.config import ReconstructionConfig
    from wt_splatting.reconstruction.pipeline import ReconstructionPipeline, ReconstructionPipelineError

    project, image_rel = _make_project_images(tmp_path, count=2)
    pipeline = ReconstructionPipeline(
        colmap_runner=_FakeColmapRunner(),
        gsplat_backend=_BadExportGsplatBackend(),
        run_id_factory=lambda: "run-bad-export",
    )

    with pytest.raises(ReconstructionPipelineError) as exc:
        pipeline.run(ReconstructionConfig.from_component_values({"image_dir": image_rel}), project)

    assert exc.value.phase == "export_splats"


def test_solver_manifest_declares_reconstruction_solver() -> None:
    manifest = tomllib.loads((ROOT / "plugins" / "wt_splatting" / "witwin.toml").read_text(encoding="utf-8"))

    solver = next(item for item in manifest["contributes"]["solvers"] if item["id"] == "witwin.splatting.reconstruct_3dgs")

    assert solver["title"] == "3DGS Reconstruction"
    assert solver["entry"] == "solver_host.py"
    assert solver["lifecycle"] == "warm"
    assert solver["maxParallel"] == 1
    assert solver["runtime"]["python"] == "${backend:python}"
    assert solver["capabilities"]["cancel"] is True
    assert solver["capabilities"]["partials"] is True


def test_solver_host_returns_serializable_result(monkeypatch, tmp_path: Path) -> None:
    import wt_splatting.solver_host as solver_host
    from wt_splatting.reconstruction.config import ReconstructionResult

    class FakePipeline:
        def run(self, config, project_root, progress):
            progress(1.0, "done")
            return ReconstructionResult(
                run_id="run-host",
                splat_path=Path(project_root) / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-host" / "model.ply",
                splat_count=9,
                summary="ok",
            )

    ctx = _SolverContext()
    monkeypatch.setattr(solver_host, "ReconstructionPipeline", lambda: FakePipeline())

    result = solver_host.solve(
        ctx,
        scene=None,
        config={
            "project_root": str(tmp_path),
            "image": {"image_dir": "assets/images"},
            "colmap": {},
            "training": {},
            "export": {},
        },
    )

    assert result == {
        "run_id": "run-host",
        "splat_path": str(tmp_path / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-host" / "model.ply"),
        "splat_count": 9,
        "format": "spark.ply",
        "summary": "ok",
    }
    assert ctx.progress_events == [(1.0, "done")]


def test_register_runtime_file_accepts_only_reconstruction_runtime(tmp_path: Path) -> None:
    from wt_splatting.io.loaders import register_3d_splats_runtime_file

    project = tmp_path / "project"
    runtime_file = project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-1" / "model.ply"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_bytes(_tiny_ply())
    outside_runtime = project / ".witwin" / "runtime" / "other" / "model.ply"
    outside_runtime.parent.mkdir(parents=True)
    outside_runtime.write_bytes(_tiny_ply())

    ref = register_3d_splats_runtime_file(BinaryAssetService(), runtime_file, project_root=project)

    assert ref.kind == "gaussian_splats"
    assert ref.format == "spark.ply"
    assert ref.attributes["splat_kind"] == "3dgs"
    with pytest.raises(ValueError):
        register_3d_splats_runtime_file(BinaryAssetService(), outside_runtime, project_root=project)


def test_component_reconstruct_calls_solver_and_binds_renderer(monkeypatch, tmp_path: Path) -> None:
    from wt_splatting.components.reconstruction import SplatReconstructionComponent
    from wt_splatting.components.splat_renderer import GaussianSplatRendererComponent

    project = tmp_path / "project"
    splat_path = project / ".witwin" / "runtime" / "wt_splatting" / "recon" / "run-component" / "model.ply"
    splat_path.parent.mkdir(parents=True)
    splat_path.write_bytes(_tiny_ply())
    owner = _ComponentOwner(project_root=project)
    reconstruction = SplatReconstructionComponent()
    renderer = GaussianSplatRendererComponent()
    owner.add_component(reconstruction)
    owner.add_component(renderer)
    reconstruction.set_field("image_dir", "assets/images", force=True)
    calls = []

    def fake_solve(solver_id, **kwargs):
        calls.append((solver_id, kwargs))
        return SimpleNamespace(
            status="succeeded",
            run_id="run-component",
            outputs={
                "result": {
                    "run_id": "run-component",
                    "splat_path": str(splat_path),
                    "splat_count": 1,
                    "format": "spark.ply",
                    "summary": "ok",
                },
            },
            error=None,
        )

    monkeypatch.setattr("wt_splatting.components.reconstruction.api.solvers.solve", fake_solve)

    message = reconstruction.click_button("reconstruct")
    values = reconstruction.to_dict()["values"]
    renderer_values = renderer.to_dict()["values"]

    assert message == "Reconstruction complete: ok"
    assert calls[0][0] == "witwin.splatting.reconstruct_3dgs"
    assert calls[0][1]["scene"] is owner.scene
    assert calls[0][1]["config"]["project_root"] == str(project)
    assert calls[0][1]["config"]["image"]["image_dir"] == "assets/images"
    assert calls[0][1]["config"]["colmap"]["matcher"] == "sequential"
    assert calls[0][1]["config"]["training"]["device"] == "cuda"
    assert calls[0][1]["config"]["export"]["export_format"] == "ply"
    assert values["status"] == "complete"
    assert values["last_run_id"] == "run-component"
    assert values["last_error"] == ""
    assert renderer_values["splats"]["kind"] == "gaussian_splats"
    assert renderer_values["splats"]["format"] == "spark.ply"
    assert renderer_values["splat_count"] == 1


@pytest.mark.requires_colmap
@pytest.mark.requires_gsplat
@pytest.mark.skipif(
    os.environ.get("WT_SPLATTING_ENABLE_REAL_RECON") != "1"
    or not os.environ.get("WT_SPLATTING_SMOKE_PROJECT")
    or not os.environ.get("WT_SPLATTING_SMOKE_IMAGE_DIR"),
    reason="real reconstruction smoke is opt-in",
)
def test_real_colmap_smoke_uses_configured_capture() -> None:
    from wt_splatting.reconstruction.config import ReconstructionConfig
    from wt_splatting.reconstruction.pipeline import ReconstructionPipeline

    project = Path(os.environ["WT_SPLATTING_SMOKE_PROJECT"])
    image_dir = os.environ["WT_SPLATTING_SMOKE_IMAGE_DIR"]
    result = ReconstructionPipeline().run(
        ReconstructionConfig.from_component_values(
            {
                "image_dir": image_dir,
                "device": os.environ.get("WT_SPLATTING_SMOKE_DEVICE", "cuda"),
                "max_steps": int(os.environ.get("WT_SPLATTING_SMOKE_MAX_STEPS", "300")),
            }
        ),
        project,
    )

    assert result.splat_path.exists()
    assert result.format == "spark.ply"


@pytest.mark.requires_gsplat
@pytest.mark.skipif(os.environ.get("WT_SPLATTING_ENABLE_REAL_RECON") != "1", reason="gsplat smoke is opt-in")
def test_real_gsplat_dependency_check() -> None:
    from wt_splatting.reconstruction.gsplat_backend import GsplatBackend

    availability = GsplatBackend().check_available()

    assert availability.available, availability.message


class _Context:
    def __init__(self, name: str) -> None:
        self.name = name
        self.position = [0.0, 0.0, 0.0]
        self.parent_id = None
        self.scene = None


class _SolverContext:
    def __init__(self) -> None:
        self.progress_events = []

    def progress(self, fraction, message) -> None:
        self.progress_events.append((fraction, message))


class _BinaryAssets:
    def __init__(self) -> None:
        self.service = BinaryAssetService()


class _ComponentServer:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.handlers = {"binary_assets": _BinaryAssets()}


class _ComponentScene:
    def __init__(self, project_root: Path) -> None:
        self._server = _ComponentServer(project_root)
        self.project_root = Path(project_root)
        self.events = []

    def _emit(self, event_type, **payload) -> None:
        self.events.append((event_type, payload))


class _ComponentOwner:
    id = "splat-recon-object"

    def __init__(self, project_root: Path) -> None:
        self.scene = _ComponentScene(project_root)
        self._components = {}

    def add_component(self, component) -> None:
        component._owner = self
        self._components[component._component_name] = component

    def get_component(self, name: str):
        return self._components.get(name)


def _make_project_images(tmp_path: Path, *, count: int) -> tuple[Path, str]:
    from PIL import Image

    project = tmp_path / "project"
    image_rel = "assets/splatting/captures/demo/images"
    image_dir = project / image_rel
    image_dir.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (4, 4), color=(index, 0, 0)).save(image_dir / f"frame_{index + 1}.jpg")
    return project, image_rel


class _FakeColmapRunner:
    def run(self, config, manifest, run_dir):
        from wt_splatting.reconstruction.colmap import ColmapRunResult

        sparse = run_dir / "colmap" / "sparse" / "0"
        dense = run_dir / "colmap" / "dense"
        sparse.mkdir(parents=True)
        dense.mkdir(parents=True)
        (sparse / "images.bin").write_bytes(b"registered")
        return ColmapRunResult(success=True, sparse_dir=run_dir / "colmap" / "sparse", dense_dir=dense, message="ok")


class _FailingColmapRunner:
    def run(self, config, manifest, run_dir):
        from wt_splatting.reconstruction.colmap import ColmapRunResult

        return ColmapRunResult(success=False, sparse_dir=run_dir / "colmap" / "sparse", dense_dir=None, message="boom")


class _FakeGsplatBackend:
    def train(self, config, colmap_result, run_dir, progress):
        from wt_splatting.reconstruction.gsplat_backend import GsplatTrainResult

        export = run_dir / "gsplat" / "exports" / "model.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(_tiny_ply())
        return GsplatTrainResult(success=True, export_path=export, message="ok")


class _BadExportGsplatBackend:
    def train(self, config, colmap_result, run_dir, progress):
        from wt_splatting.reconstruction.gsplat_backend import GsplatTrainResult

        export = run_dir / "gsplat" / "exports" / "model.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"not a ply")
        return GsplatTrainResult(success=True, export_path=export, message="bad")


def _tiny_ply() -> bytes:
    return (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"end_header\n"
        b"0 0 0 255 0 0\n"
    )
