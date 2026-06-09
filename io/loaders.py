from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from witwin_server.features.assets.binary.refs import RenderResourceRef
from witwin_server.features.assets.binary.service import BinaryAssetService

from .formats import (
    GAUSSIAN_SPLATS_2D_KIND,
    GAUSSIAN_SPLATS_KIND,
    WITWIN_2DGS_QUAD_FORMAT,
    encode_2d_gaussian_splats,
)


SPARK_FORMATS = {
    ".ply": "spark.ply",
    ".spz": "spark.spz",
    ".splat": "spark.splat",
    ".ksplat": "spark.ksplat",
    ".sog": "spark.sog",
    ".zip": "spark.zip",
    ".rad": "spark.rad",
}


def register_3d_splats_file(
    service: BinaryAssetService,
    path: str | Path,
    *,
    scene_id: str = "default",
    project_root: str | Path | None = None,
) -> RenderResourceRef:
    source = _resolve_project_path(path, project_root)
    suffix = source.suffix.lower()
    if suffix not in SPARK_FORMATS:
        raise ValueError(f"Unsupported Spark splat format: {suffix}")
    data = source.read_bytes()
    return service.register_bytes(
        kind=GAUSSIAN_SPLATS_KIND,
        format=SPARK_FORMATS[suffix],
        data=data,
        scene_id=scene_id,
        shape=[_estimate_ply_vertex_count(data) if suffix == ".ply" else 0],
        attributes={"splat_kind": "3dgs", "file_name": source.name},
        asset_id=source.name,
    )


def register_3d_splats_runtime_file(
    service: BinaryAssetService,
    path: str | Path,
    *,
    scene_id: str = "default",
    project_root: str | Path | None = None,
    asset_id: str | None = None,
) -> RenderResourceRef:
    source = _resolve_reconstruction_runtime_path(path, project_root)
    suffix = source.suffix.lower()
    if suffix not in SPARK_FORMATS:
        raise ValueError(f"Unsupported Spark splat format: {suffix}")
    data = source.read_bytes()
    return service.register_bytes(
        kind=GAUSSIAN_SPLATS_KIND,
        format=SPARK_FORMATS[suffix],
        data=data,
        scene_id=scene_id,
        shape=[_estimate_ply_vertex_count(data) if suffix == ".ply" else 0],
        attributes={"splat_kind": "3dgs", "file_name": source.name},
        asset_id=asset_id or source.name,
    )


def register_2d_splats_arrays(
    service: BinaryAssetService,
    *,
    centers: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    sigma_uv: np.ndarray,
    opacity: np.ndarray,
    color_dc: np.ndarray,
    sh_degree: int = 0,
    attributes: dict[str, Any] | None = None,
    scene_id: str = "default",
    asset_id: str = "demo-2dgs.w2gs",
) -> RenderResourceRef:
    data = encode_2d_gaussian_splats(
        centers=centers,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        sigma_uv=sigma_uv,
        opacity=opacity,
        color_dc=color_dc,
        sh_degree=sh_degree,
        attributes={"splat_kind": "2dgs", **(attributes or {})},
    )
    return service.register_bytes(
        kind=GAUSSIAN_SPLATS_2D_KIND,
        format=WITWIN_2DGS_QUAD_FORMAT,
        data=data,
        scene_id=scene_id,
        shape=[int(np.asarray(centers).shape[0])],
        attributes={"splat_kind": "2dgs", "splat_count": int(np.asarray(centers).shape[0])},
        asset_id=asset_id,
    )


def _resolve_project_path(path: str | Path, project_root: str | Path | None) -> Path:
    candidate = Path(path)
    if project_root is None:
        raise ValueError("project_root is required when registering splat files")
    root = Path(project_root).resolve(strict=True)
    resolved = (root / candidate).resolve(strict=True) if not candidate.is_absolute() else candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Splat file must be inside the Project: {resolved}") from exc
    return resolved


def _resolve_reconstruction_runtime_path(path: str | Path, project_root: str | Path | None) -> Path:
    if project_root is None:
        raise ValueError("project_root is required when registering runtime splat files")
    root = Path(project_root).resolve(strict=True)
    candidate = Path(path)
    resolved = (root / candidate).resolve(strict=True) if not candidate.is_absolute() else candidate.resolve(strict=True)
    runtime_root = (root / ".witwin" / "runtime" / "wt_splatting" / "recon").resolve()
    try:
        resolved.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(f"Runtime splat file must be under {runtime_root}: {resolved}") from exc
    return resolved


def _estimate_ply_vertex_count(data: bytes) -> int:
    header = data[:4096].decode("utf-8", errors="ignore")
    match = re.search(r"^element\s+vertex\s+(\d+)\s*$", header, flags=re.MULTILINE)
    return int(match.group(1)) if match else 0
