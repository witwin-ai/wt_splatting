from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .colmap import ColmapRunner
from .config import ReconstructionConfig, ReconstructionResult
from .export import validate_spark_export
from .gsplat_backend import GsplatBackend
from .images import build_image_manifest


ProgressCallback = Callable[[float, str], None]


class ReconstructionPipelineError(RuntimeError):
    def __init__(self, run_id: str, phase: str, message: str) -> None:
        self.run_id = run_id
        self.phase = phase
        self.message = message
        super().__init__(f"{phase} failed for {run_id}: {message}")


class ReconstructionPipeline:
    def __init__(
        self,
        *,
        colmap_runner: ColmapRunner | None = None,
        gsplat_backend: GsplatBackend | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._colmap_runner = colmap_runner or ColmapRunner()
        self._gsplat_backend = gsplat_backend or GsplatBackend()
        self._run_id_factory = run_id_factory or (lambda: uuid4().hex)

    def run(
        self,
        config: ReconstructionConfig,
        project_root: Path,
        progress: ProgressCallback | None = None,
    ) -> ReconstructionResult:
        run_id = self._run_id_factory()
        project_root = Path(project_root).resolve()
        run_dir = project_root / ".witwin" / "runtime" / "wt_splatting" / "recon" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        emit = progress or (lambda *_: None)
        phase = "validate_images"

        try:
            phase = "validate_images"
            emit(0.05, "validate images")
            manifest = build_image_manifest(project_root, config.image, run_dir)

            phase = "run_colmap"
            emit(0.25, "build COLMAP model")
            colmap_result = self._colmap_runner.run(config.colmap, manifest, run_dir)
            if not colmap_result.success:
                raise ReconstructionPipelineError(run_id, "run_colmap", colmap_result.message)

            phase = "train_gsplat"
            emit(0.60, "train gsplat model")
            train_result = self._gsplat_backend.train(config.training, colmap_result, run_dir, emit)
            if not train_result.success or train_result.export_path is None:
                raise ReconstructionPipelineError(run_id, "train_gsplat", train_result.message)

            phase = "export_splats"
            emit(0.90, "export splats")
            final_export = self._final_export_path(train_result.export_path, config)
            exported = validate_spark_export(final_export)

            result = ReconstructionResult(
                run_id=run_id,
                splat_path=exported.path,
                splat_count=exported.splat_count,
                format=exported.format,
                summary=f"Reconstructed {exported.splat_count} splats from {manifest.image_count} images",
            )
            phase = "write_result"
            emit(1.0, "write result")
            self._write_result(run_dir, result, config)
            return result
        except ReconstructionPipelineError:
            raise
        except Exception as exc:
            raise ReconstructionPipelineError(run_id, phase, str(exc)) from exc

    def _final_export_path(self, export_path: Path, config: ReconstructionConfig) -> Path:
        source = Path(export_path)
        target = source.with_name(f"{config.export.output_name}.ply")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def _write_result(self, run_dir: Path, result: ReconstructionResult, config: ReconstructionConfig) -> None:
        payload = {
            "result": result.to_dict(),
            "config": config.to_dict(),
        }
        (run_dir / "pipeline_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
