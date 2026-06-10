from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .backends import SuGaRBackend, TwoDGSBackend
from .backends.base import BackendMeshResult
from .config import MeshizationConfig
from .mesh import extract_density_mesh
from .ply import GaussianSplatData, read_gaussian_ply, write_mesh_ply


MeshExtractor = Callable[[GaussianSplatData, MeshizationConfig], tuple[object, object, object]]
MeshBackend = object


@dataclass(frozen=True)
class MeshizationResult:
    run_id: str
    run_dir: Path
    mesh_path: Path
    vertex_count: int
    face_count: int
    format: str
    summary: str


class MeshizationPipeline:
    def __init__(self, extractor: MeshExtractor | None = None, backends: dict[str, MeshBackend] | None = None) -> None:
        self._extractor = extractor or extract_density_mesh
        self._backends = backends or {"sugar": SuGaRBackend(), "2dgs": TwoDGSBackend()}

    def run(
        self,
        config: MeshizationConfig,
        project_root: str | Path,
        progress: Callable[[float, str], None] | None = None,
    ) -> MeshizationResult:
        project = Path(project_root).resolve()
        run_id = uuid4().hex
        run_dir = project / ".witwin" / "runtime" / "wt_splatting" / "mesh" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if config.method == "density":
            _progress(progress, 0.10, "load splats")
            splats = read_gaussian_ply(config.splat_path)
            _progress(progress, 0.35, "extract density mesh")
            vertices, faces, colors = self._extractor(splats, config)
            _progress(progress, 0.80, "write mesh")
            mesh_path = run_dir / f"{config.output_name}.ply"
            write_mesh_ply(mesh_path, vertices, faces, colors)
            vertex_count = int(len(vertices))
            face_count = int(len(faces))
            summary = f"{vertex_count} vertices, {face_count} faces"
        else:
            backend = self._backends.get(config.method)
            if backend is None or not hasattr(backend, "run"):
                raise ValueError(f"No meshization backend registered for method {config.method}")
            _progress(progress, 0.10, f"start {config.method}")
            backend_result = backend.run(config, run_dir, progress)
            if not isinstance(backend_result, BackendMeshResult):
                raise TypeError(f"{config.method} backend returned an invalid result")
            mesh_path = backend_result.mesh_path
            vertex_count = backend_result.vertex_count
            face_count = backend_result.face_count
            summary = backend_result.summary

        result = MeshizationResult(
            run_id=run_id,
            run_dir=run_dir,
            mesh_path=mesh_path,
            vertex_count=vertex_count,
            face_count=face_count,
            format="mesh.ply",
            summary=summary,
        )
        (run_dir / "meshization_result.json").write_text(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "mesh_path": str(result.mesh_path),
                    "vertex_count": result.vertex_count,
                    "face_count": result.face_count,
                    "format": result.format,
                    "summary": result.summary,
                    "method": config.method,
                    "source_path": str(config.source_path),
                    "grid_resolution": config.grid_resolution,
                    "iso_level": config.iso_level,
                    "opacity_threshold": config.opacity_threshold,
                    "max_gaussians": config.max_gaussians,
                    "sigma_radius": config.sigma_radius,
                    "full_convergence_steps": config.full_convergence_steps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _progress(progress, 1.00, "complete")
        return result


def _progress(callback: Callable[[float, str], None] | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(fraction, message)
