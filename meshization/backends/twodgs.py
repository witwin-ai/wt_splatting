from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Callable

from wt_splatting.reconstruction.colmap import ColmapRunner
from wt_splatting.reconstruction.config import ColmapConfig, ImageSequenceConfig
from wt_splatting.reconstruction.images import build_image_manifest, discover_images

from ..config import MeshizationConfig
from .base import ExternalMeshBackend


class TwoDGSBackend(ExternalMeshBackend):
    method = "2dgs"
    script_env = "WT_SPLATTING_2DGS_SCRIPT"

    def __init__(self, colmap_runner: ColmapRunner | None = None) -> None:
        self._colmap_runner = colmap_runner or ColmapRunner()

    def run(
        self,
        config: MeshizationConfig,
        run_dir: Path,
        progress: Callable[[float, str], None] | None = None,
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        prepared_source = self._prepare_source(config, run_dir, progress)
        prepared_config = replace(config, source_path=prepared_source)
        return self._run_external(prepared_config, run_dir, progress)

    def _run_external(self, prepared_config: MeshizationConfig, run_dir: Path, progress: Callable[[float, str], None] | None = None):
        return super().run(prepared_config, run_dir, progress)

    def _script(self, config):
        if config.backend_script is not None or os.environ.get(self.script_env, "").strip():
            return super()._script(config)
        return Path(__file__).resolve().parents[2] / "tools" / "adapters" / "twodgs_official.py"

    def build_command(self, config: MeshizationConfig, run_dir: Path) -> list[str]:
        command = super().build_command(config, run_dir)
        if config.backend_script is None and not os.environ.get(self.script_env, "").strip():
            command.extend(["--mesh-res", str(config.mesh_resolution), "--mesh-cluster-count", str(config.mesh_cluster_count)])
        return command

    def _prepare_source(
        self,
        config: MeshizationConfig,
        run_dir: Path,
        progress: Callable[[float, str], None] | None,
    ) -> Path:
        source = config.source_path
        if _is_2dgs_scene(source):
            return source
        if not source.is_dir() or not discover_images(source, max_images=1):
            raise ValueError(f"2DGS source_path must be a COLMAP scene or image directory: {source}")
        if progress is not None:
            progress(0.18, "prepare COLMAP scene for 2DGS")
        try:
            relative_image_dir = source.resolve().relative_to(config.project_root)
        except ValueError as exc:
            raise ValueError(f"2DGS image directory must stay inside the Project: {source}") from exc
        image_config = ImageSequenceConfig(
            image_dir=str(relative_image_dir).replace("\\", "/"),
            capture_id=f"2dgs-{config.output_name}",
            downscale_factor=config.downscale_factor,
            max_images=config.max_images,
        )
        manifest = build_image_manifest(config.project_root, image_config, run_dir)
        colmap_result = self._colmap_runner.run(
            ColmapConfig(
                colmap_binary=config.colmap_binary,
                camera_model=config.camera_model,
                matcher=config.matcher,
            ),
            manifest,
            run_dir,
        )
        if not colmap_result.success or colmap_result.dense_dir is None:
            raise RuntimeError(f"COLMAP prepare for 2DGS failed: {colmap_result.message}")
        normalized = _normalize_colmap_scene(colmap_result.dense_dir, run_dir)
        if not _is_2dgs_scene(normalized):
            raise RuntimeError(f"COLMAP output is not a 2DGS-compatible scene: {normalized}")
        return normalized


def _is_2dgs_scene(path: Path) -> bool:
    return (path / "sparse").exists() or (path / "transforms_train.json").is_file()


def _normalize_colmap_scene(source: Path, run_dir: Path) -> Path:
    source = Path(source)
    if (source / "sparse" / "0").is_dir():
        return source
    sparse = source / "sparse"
    if not sparse.is_dir() or not (sparse / "images.bin").exists():
        return source
    target = Path(run_dir) / "2dgs_scene"
    target_sparse = target / "sparse" / "0"
    target_images = target / "images"
    target_sparse.mkdir(parents=True, exist_ok=True)
    for item in sparse.iterdir():
        if item.is_file():
            shutil.copy2(item, target_sparse / item.name)
    images = source / "images"
    if images.is_dir():
        shutil.copytree(images, target_images, dirs_exist_ok=True)
    return target
