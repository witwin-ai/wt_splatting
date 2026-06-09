from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportedSplat:
    path: Path
    format: str
    splat_count: int


def validate_spark_export(path: Path) -> ExportedSplat:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Splat export does not exist: {source}")
    if source.suffix.lower() != ".ply":
        raise ValueError(f"Only .ply exports are supported in V1: {source.suffix}")
    data = source.read_bytes()
    if not data.startswith(b"ply"):
        raise ValueError(f"Export is not a PLY file: {source}")
    return ExportedSplat(path=source, format="spark.ply", splat_count=_estimate_ply_vertex_count(data))


def _estimate_ply_vertex_count(data: bytes) -> int:
    header = data[:4096].decode("utf-8", errors="ignore")
    match = re.search(r"^element\s+vertex\s+(\d+)\s*$", header, flags=re.MULTILINE)
    return int(match.group(1)) if match else 0
