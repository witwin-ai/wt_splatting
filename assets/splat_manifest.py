from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WTSPLAT_FORMAT = "witwin.splat_asset.v1"
WTSPLAT_EXT = ".wtsplat"
THREEDGS_PLY_FORMAT = "3dgs.ply"

_THREEDGS_REQUIRED = {
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


@dataclass(frozen=True)
class PLYClassification:
    kind: str
    vertex_count: int = 0
    splat_count: int = 0
    ambiguous: bool = False


@dataclass(frozen=True)
class WTSplatManifest:
    path: Path
    source_path: Path
    source_rel: str
    format: str
    splat_count: int = 0


def classify_ply(path: str | Path) -> PLYClassification:
    source = Path(path)
    header = _read_ply_header(source)
    vertex_count = 0
    vertex_props: set[str] = set()
    has_face = False
    in_vertex = False

    for raw in header:
        tokens = raw.strip().split()
        if not tokens:
            continue
        if len(tokens) >= 3 and tokens[0] == "element":
            in_vertex = tokens[1] == "vertex"
            if tokens[1] == "vertex":
                vertex_count = int(tokens[2])
            elif tokens[1] == "face":
                has_face = True
            continue
        if in_vertex and len(tokens) >= 3 and tokens[0] == "property" and tokens[1] != "list":
            vertex_props.add(tokens[2])

    has_3dgs_shape = _THREEDGS_REQUIRED.issubset(vertex_props)
    has_color = {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(vertex_props) or {"red", "green", "blue"}.issubset(vertex_props)
    is_3dgs = has_3dgs_shape and has_color
    if is_3dgs:
        return PLYClassification(
            kind="gaussian_splat_3dgs",
            vertex_count=vertex_count,
            splat_count=vertex_count,
            ambiguous=has_face,
        )
    if has_face:
        return PLYClassification(kind="mesh_ply", vertex_count=vertex_count)
    return PLYClassification(kind="point_ply", vertex_count=vertex_count)


def write_wtsplat_manifest(path: str | Path, source_path: str | Path, *, project_root: str | Path) -> WTSplatManifest:
    manifest_path = Path(path).resolve()
    root = Path(project_root).resolve()
    source = _resolve_inside_project(Path(source_path), root)
    classification = classify_ply(source)
    if classification.kind != "gaussian_splat_3dgs":
        raise ValueError(f"Expected a 3DGS PLY source, got {classification.kind}")
    source_rel = _project_rel(source, root)
    payload = {
        "format": WTSPLAT_FORMAT,
        "source": source_rel,
        "splat_format": THREEDGS_PLY_FORMAT,
        "splat_count": classification.splat_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return WTSplatManifest(
        path=manifest_path,
        source_path=source,
        source_rel=source_rel,
        format=THREEDGS_PLY_FORMAT,
        splat_count=classification.splat_count,
    )


def read_wtsplat_manifest(path: str | Path, *, project_root: str | Path) -> WTSplatManifest:
    manifest_path = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid .wtsplat JSON: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(".wtsplat manifest must be a JSON object")
    source_value = str(data.get("source") or "").strip()
    if not source_value:
        raise ValueError(".wtsplat manifest requires source")
    source = _resolve_manifest_source(source_value, manifest_path, root)
    if not source.is_file():
        raise ValueError(f"3DGS source does not exist: {source}")
    splat_format = str(data.get("splat_format") or data.get("format") or THREEDGS_PLY_FORMAT)
    if splat_format == WTSPLAT_FORMAT:
        splat_format = THREEDGS_PLY_FORMAT
    if splat_format != THREEDGS_PLY_FORMAT:
        raise ValueError(f"Unsupported splat format: {splat_format}")
    classification = classify_ply(source)
    if classification.kind != "gaussian_splat_3dgs":
        raise ValueError(f".wtsplat source is not a 3DGS PLY: {classification.kind}")
    return WTSplatManifest(
        path=manifest_path,
        source_path=source,
        source_rel=_project_rel(source, root),
        format=splat_format,
        splat_count=int(data.get("splat_count") or classification.splat_count),
    )


def _read_ply_header(path: Path) -> list[str]:
    lines: list[bytes] = []
    with path.open("rb") as handle:
        for _ in range(512):
            line = handle.readline()
            if not line:
                break
            lines.append(line)
            if line.rstrip(b"\r\n") == b"end_header":
                break
    if not lines or lines[0].strip() != b"ply":
        raise ValueError(f"Not a PLY file: {path}")
    if not lines[-1].rstrip(b"\r\n") == b"end_header":
        raise ValueError(f"PLY header is missing end_header: {path}")
    return [line.decode("ascii", errors="ignore").strip() for line in lines]


def _resolve_manifest_source(source: str, manifest_path: Path, root: Path) -> Path:
    candidate = Path(source)
    if candidate.is_absolute():
        return _resolve_inside_project(candidate, root)
    local = (manifest_path.parent / candidate).resolve()
    try:
        local.relative_to(root)
        if local.exists():
            return local
    except ValueError:
        pass
    return _resolve_inside_project(root / source.replace("\\", "/").lstrip("/"), root)


def _resolve_inside_project(path: Path, root: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Splat asset source must stay inside the Project") from exc
    return target


def _project_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()
