from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_METHODS = {"density", "sugar", "2dgs"}

@dataclass(frozen=True)
class MeshizationConfig:
    project_root: Path
    source_path: Path
    method: str = "density"
    output_name: str = "splat_mesh"
    grid_resolution: int = 64
    iso_level: float = 0.15
    opacity_threshold: float = 0.01
    max_gaussians: int = 50_000
    sigma_radius: float = 3.0
    export_format: str = "ply"
    backend_script: Path | None = None
    backend_repo: Path | None = None
    backend_python: Path | None = None
    full_convergence_steps: int = 30_000
    colmap_binary: str = "colmap"
    camera_model: str = "OPENCV"
    matcher: str = "sequential"
    downscale_factor: int = 1
    max_images: int = 0
    mesh_resolution: int = 1024
    mesh_cluster_count: int = 1

    @property
    def splat_path(self) -> Path:
        return self.source_path

    @classmethod
    def from_component_values(cls, values: dict[str, Any], *, project_root: str | Path) -> "MeshizationConfig":
        root = Path(project_root).resolve()
        method = str(values.get("method") or "density").lower()
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {', '.join(sorted(VALID_METHODS))}")
        source_path = _resolve_project_source(
            root,
            values.get("source_path") or values.get("splat_path") or values.get("scene_path") or values.get("image_dir") or "",
            require_ply=method == "density",
        )
        output_name = _safe_name(str(values.get("output_name") or "splat_mesh"))
        grid_resolution = _int_value(values.get("grid_resolution", 64), "grid_resolution", min_value=8, max_value=256)
        iso_level = _float_value(values.get("iso_level", 0.15), "iso_level", min_value=0.0)
        opacity_threshold = _float_value(values.get("opacity_threshold", 0.01), "opacity_threshold", min_value=0.0, max_value=1.0)
        max_gaussians = _int_value(values.get("max_gaussians", 50_000), "max_gaussians", min_value=1)
        sigma_radius = _float_value(values.get("sigma_radius", 3.0), "sigma_radius", min_value=0.25, max_value=8.0)
        export_format = str(values.get("export_format") or "ply").lower()
        if export_format != "ply":
            raise ValueError(f"Unsupported mesh export_format: {export_format}")
        backend_script = _optional_path(values.get("backend_script"))
        backend_repo = _optional_path(values.get("backend_repo"))
        backend_python = _optional_path(values.get("backend_python"))
        full_convergence_steps = _int_value(values.get("full_convergence_steps", 30_000), "full_convergence_steps", min_value=1)
        colmap_binary = str(values.get("colmap_binary") or "colmap").strip()
        if not colmap_binary:
            raise ValueError("colmap_binary is required")
        camera_model = str(values.get("camera_model") or "OPENCV").strip()
        if not camera_model:
            raise ValueError("camera_model is required")
        matcher = str(values.get("matcher") or "sequential").strip()
        if matcher not in {"sequential", "exhaustive"}:
            raise ValueError("matcher must be sequential or exhaustive")
        downscale_factor = _int_value(values.get("downscale_factor", 1), "downscale_factor", min_value=1)
        if downscale_factor not in {1, 2, 4, 8}:
            raise ValueError("downscale_factor must be one of 1, 2, 4, or 8")
        max_images = _int_value(values.get("max_images", 0), "max_images", min_value=0)
        mesh_resolution = _int_value(values.get("mesh_resolution", 1024), "mesh_resolution", min_value=512, max_value=4096)
        if mesh_resolution % 512 != 0:
            raise ValueError("mesh_resolution must be a multiple of 512 for the built-in 2DGS adapter")
        mesh_cluster_count = _int_value(values.get("mesh_cluster_count", 1), "mesh_cluster_count", min_value=1, max_value=1000)
        return cls(
            project_root=root,
            source_path=source_path,
            method=method,
            output_name=output_name,
            grid_resolution=grid_resolution,
            iso_level=iso_level,
            opacity_threshold=opacity_threshold,
            max_gaussians=max_gaussians,
            sigma_radius=sigma_radius,
            export_format=export_format,
            backend_script=backend_script,
            backend_repo=backend_repo,
            backend_python=backend_python,
            full_convergence_steps=full_convergence_steps,
            colmap_binary=colmap_binary,
            camera_model=camera_model,
            matcher=matcher,
            downscale_factor=downscale_factor,
            max_images=max_images,
            mesh_resolution=mesh_resolution,
            mesh_cluster_count=mesh_cluster_count,
        )


def _resolve_project_source(project_root: Path, value: str | Path, *, require_ply: bool) -> Path:
    if not value:
        raise ValueError("source_path is required")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"source_path must stay inside the Project: {resolved}") from exc
    if not resolved.exists():
        raise ValueError(f"source_path does not exist: {resolved}")
    if require_ply and not resolved.is_file():
        raise ValueError(f"Density meshization requires a .ply file source_path: {resolved}")
    if require_ply and resolved.suffix.lower() != ".ply":
        raise ValueError(f"Only .ply 3DGS inputs are supported for meshization: {resolved.suffix}")
    return resolved


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).resolve()


def _safe_name(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return token or "splat_mesh"


def _int_value(value: Any, name: str, *, min_value: int, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return parsed


def _float_value(value: Any, name: str, *, min_value: float, max_value: float | None = None) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return parsed
