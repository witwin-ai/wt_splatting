from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeshizationConfig:
    splat_path: Path
    output_name: str = "splat_mesh"
    grid_resolution: int = 64
    iso_level: float = 0.15
    opacity_threshold: float = 0.01
    max_gaussians: int = 50_000
    sigma_radius: float = 3.0
    export_format: str = "ply"

    @classmethod
    def from_component_values(cls, values: dict[str, Any], *, project_root: str | Path) -> "MeshizationConfig":
        root = Path(project_root).resolve()
        splat_path = _resolve_project_file(root, values.get("splat_path") or values.get("source_path") or "")
        output_name = _safe_name(str(values.get("output_name") or "splat_mesh"))
        grid_resolution = _int_value(values.get("grid_resolution", 64), "grid_resolution", min_value=8, max_value=256)
        iso_level = _float_value(values.get("iso_level", 0.15), "iso_level", min_value=0.0)
        opacity_threshold = _float_value(values.get("opacity_threshold", 0.01), "opacity_threshold", min_value=0.0, max_value=1.0)
        max_gaussians = _int_value(values.get("max_gaussians", 50_000), "max_gaussians", min_value=1)
        sigma_radius = _float_value(values.get("sigma_radius", 3.0), "sigma_radius", min_value=0.25, max_value=8.0)
        export_format = str(values.get("export_format") or "ply").lower()
        if export_format != "ply":
            raise ValueError(f"Unsupported mesh export_format: {export_format}")
        return cls(
            splat_path=splat_path,
            output_name=output_name,
            grid_resolution=grid_resolution,
            iso_level=iso_level,
            opacity_threshold=opacity_threshold,
            max_gaussians=max_gaussians,
            sigma_radius=sigma_radius,
            export_format=export_format,
        )


def _resolve_project_file(project_root: Path, value: str | Path) -> Path:
    if not value:
        raise ValueError("splat_path is required")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"splat_path must stay inside the Project: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"splat_path does not exist: {resolved}")
    if resolved.suffix.lower() != ".ply":
        raise ValueError(f"Only .ply 3DGS inputs are supported for meshization: {resolved.suffix}")
    return resolved


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
