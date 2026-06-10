from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from wt_splatting.meshization.config import MeshizationConfig
from wt_splatting.meshization.pipeline import MeshizationPipeline
from wt_splatting.reconstruction.images import SUPPORTED_IMAGE_SUFFIXES

DATASET_ID = "nvs-bench/mipnerf360"
DEFAULT_SCENES = ("kitchen", "room")
DEFAULT_METHODS = ("sugar", "2dgs")


@dataclass(frozen=True)
class StagedScene:
    scene: str
    scene_dir: Path
    image_count: int


@dataclass(frozen=True)
class BenchmarkRecord:
    scene: str
    method: str
    status: str
    run_id: str = ""
    mesh_path: str = ""
    vertex_count: int = 0
    face_count: int = 0
    elapsed_sec: float = 0.0
    message: str = ""


def resolve_dataset_root(dataset_root: str | Path | None = None, *, allow_download: bool = True) -> Path:
    if dataset_root:
        root = Path(dataset_root).resolve()
        if not root.exists():
            raise ValueError(f"dataset_root does not exist: {root}")
        return root
    env_root = os.environ.get("WT_SPLATTING_MIPNERF360_ROOT", "").strip()
    if env_root:
        return resolve_dataset_root(env_root, allow_download=False)
    if not allow_download:
        raise ValueError("dataset_root is required when downloads are disabled")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download nvs-bench/mipnerf360") from exc
    return Path(snapshot_download(repo_id=DATASET_ID, repo_type="dataset")).resolve()


def discover_scene_images(dataset_root: str | Path, scene: str) -> list[Path]:
    root = Path(dataset_root).resolve()
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    scene_lower = scene.lower()
    matching = [path for path in candidates if scene_lower in [part.lower() for part in path.parts]]
    if not matching:
        matching = [path for path in candidates if scene_lower in path.as_posix().lower()]
    matching = sorted(matching, key=_natural_key)
    if not matching:
        raise ValueError(f"No {scene} images found under {root}")
    return matching


def stage_scene_images(images: Iterable[Path], project_root: str | Path, scene: str, *, max_images: int = 0) -> StagedScene:
    selected = list(images)
    if max_images > 0:
        selected = selected[:max_images]
    if not selected:
        raise ValueError(f"No images to stage for {scene}")
    project = Path(project_root).resolve()
    scene_dir = project / ".witwin" / "runtime" / "wt_splatting" / "benchmarks" / "mipnerf360" / scene / "input"
    scene_dir.mkdir(parents=True, exist_ok=True)
    _clear_staged_scene(scene_dir)
    for index, source in enumerate(selected, start=1):
        target = scene_dir / f"{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, target)
    manifest = {
        "scene": scene,
        "image_count": len(selected),
        "source_images": [str(path) for path in selected],
        "staged_dir": str(scene_dir),
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return StagedScene(scene=scene, scene_dir=scene_dir, image_count=len(selected))


def run_benchmark(
    *,
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    scenes: Iterable[str] = DEFAULT_SCENES,
    methods: Iterable[str] = DEFAULT_METHODS,
    allow_download: bool = True,
    max_images: int = 0,
    full_convergence_steps: int = 30_000,
    backend_python: str | Path | None = None,
    sugar_script: str | Path | None = None,
    twodgs_script: str | Path | None = None,
    sugar_repo: str | Path | None = None,
    twodgs_repo: str | Path | None = None,
    colmap_binary: str | Path | None = None,
    camera_model: str = "OPENCV",
    matcher: str = "sequential",
    downscale_factor: int = 1,
    mesh_resolution: int = 1024,
    mesh_cluster_count: int = 1,
) -> list[BenchmarkRecord]:
    project = Path(project_root).resolve()
    root = resolve_dataset_root(dataset_root, allow_download=allow_download)
    records: list[BenchmarkRecord] = []
    for scene in scenes:
        staged = stage_scene_images(discover_scene_images(root, scene), project, scene, max_images=max_images)
        for method in methods:
            script = sugar_script if method == "sugar" else twodgs_script if method == "2dgs" else None
            repo = _method_repo(method, sugar_repo=sugar_repo, twodgs_repo=twodgs_repo)
            started = time.perf_counter()
            try:
                config = MeshizationConfig.from_component_values(
                    {
                        "method": method,
                        "source_path": str(staged.scene_dir),
                        "output_name": f"{scene}_{method}_mesh",
                        "backend_script": str(script or ""),
                        "backend_repo": str(repo or ""),
                        "backend_python": str(backend_python or ""),
                        "full_convergence_steps": full_convergence_steps,
                        "colmap_binary": str(colmap_binary or "colmap"),
                        "camera_model": camera_model,
                        "matcher": matcher,
                        "downscale_factor": downscale_factor,
                        "mesh_resolution": mesh_resolution,
                        "mesh_cluster_count": mesh_cluster_count,
                    },
                    project_root=project,
                )
                result = MeshizationPipeline().run(config, project)
                records.append(
                    BenchmarkRecord(
                        scene=scene,
                        method=method,
                        status="succeeded",
                        run_id=result.run_id,
                        mesh_path=str(result.mesh_path),
                        vertex_count=result.vertex_count,
                        face_count=result.face_count,
                        elapsed_sec=time.perf_counter() - started,
                        message=result.summary,
                    )
                )
            except Exception as exc:
                records.append(
                    BenchmarkRecord(
                        scene=scene,
                        method=method,
                        status="failed",
                        elapsed_sec=time.perf_counter() - started,
                        message=str(exc),
                    )
                )
    write_report(project / ".witwin" / "runtime" / "wt_splatting" / "benchmarks" / "mipnerf360", records)
    return records


def write_report(root: str | Path, records: list[BenchmarkRecord]) -> Path:
    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_result.json"
    json_path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    report_path = out_dir / "benchmark_report.md"
    lines = ["# Mip-NeRF 360 Splatting Mesh Benchmark", "", "| Scene | Method | Status | Vertices | Faces | Seconds | Message |", "|---|---|---|---:|---:|---:|---|"]
    for record in records:
        lines.append(
            f"| {record.scene} | {record.method} | {record.status} | {record.vertex_count} | {record.face_count} | {record.elapsed_sec:0.3f} | {_escape(record.message)} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _clear_staged_scene(scene_dir: Path) -> None:
    for path in scene_dir.iterdir():
        if path.is_file() and (path.name == "manifest.json" or path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES):
            path.unlink()


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _method_repo(method: str, *, sugar_repo: str | Path | None, twodgs_repo: str | Path | None) -> str | Path | None:
    if method == "sugar":
        return sugar_repo or os.environ.get("WT_SPLATTING_SUGAR_REPO", "").strip()
    if method == "2dgs":
        return twodgs_repo or os.environ.get("WT_SPLATTING_2DGS_REPO", "").strip()
    return None
