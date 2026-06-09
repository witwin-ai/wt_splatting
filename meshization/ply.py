from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianSplatData:
    centers: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])


def read_gaussian_ply(path: str | Path) -> GaussianSplatData:
    source = Path(path)
    data = source.read_bytes()
    header_end = data.find(b"end_header\n")
    newline_len = 1
    if header_end < 0:
        header_end = data.find(b"end_header\r\n")
        newline_len = 2
    if header_end < 0:
        raise ValueError(f"PLY header is missing end_header: {source}")
    header_bytes = data[: header_end + len("end_header") + newline_len]
    header = header_bytes.decode("ascii", errors="ignore").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {source}")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary_little_endian PLY 3DGS inputs are supported")

    vertex_count, properties = _vertex_layout(header)
    if vertex_count <= 0:
        raise ValueError("PLY contains no Gaussian vertices")
    row_format = "<" + "".join(_STRUCT_FORMAT[prop_type] for prop_type, _ in properties)
    row_size = struct.calcsize(row_format)
    payload = data[len(header_bytes) :]
    expected = vertex_count * row_size
    if len(payload) < expected:
        raise ValueError(f"PLY vertex payload is truncated: expected {expected} bytes, found {len(payload)}")

    columns = {name: np.empty(vertex_count, dtype=np.float32) for _, name in properties}
    for index in range(vertex_count):
        values = struct.unpack_from(row_format, payload, index * row_size)
        for (_, name), value in zip(properties, values):
            columns[name][index] = float(value)

    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"PLY missing required Gaussian properties: {', '.join(missing)}")

    centers = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float32)
    raw_scales = np.stack([columns["scale_0"], columns["scale_1"], columns["scale_2"]], axis=1).astype(np.float32)
    scales = np.exp(np.clip(raw_scales, -12.0, 8.0)).astype(np.float32)
    raw_opacity = columns["opacity"].astype(np.float32)
    opacities = _normalize_opacity(raw_opacity)
    colors = _decode_color(columns)
    return GaussianSplatData(centers=centers, scales=scales, opacities=opacities, colors=colors)


def write_mesh_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if len(vertices) and (int(faces.min(initial=0)) < 0 or int(faces.max(initial=0)) >= len(vertices)):
        raise ValueError("faces contain vertex indices outside the vertex array")
    if colors is None:
        colors_u8 = np.full((len(vertices), 3), 220, dtype=np.uint8)
    else:
        colors_u8 = np.clip(np.rint(np.asarray(colors, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
        if colors_u8.shape != (len(vertices), 3):
            raise ValueError("colors must have shape (N, 3) matching vertices")

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with target.open("wb") as handle:
        handle.write(header)
        for vertex, color in zip(vertices, colors_u8):
            handle.write(struct.pack("<3f3B", float(vertex[0]), float(vertex[1]), float(vertex[2]), int(color[0]), int(color[1]), int(color[2])))
        for face in faces:
            handle.write(struct.pack("<B3i", 3, int(face[0]), int(face[1]), int(face[2])))


def _vertex_layout(header: list[str]) -> tuple[int, list[tuple[str, str]]]:
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for raw in header:
        line = raw.strip()
        if line.startswith("element "):
            tokens = line.split()
            in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
            if in_vertex:
                vertex_count = int(tokens[2])
            continue
        if in_vertex and line.startswith("property "):
            tokens = line.split()
            if len(tokens) == 3 and tokens[1] in _STRUCT_FORMAT:
                properties.append((tokens[1], tokens[2]))
    return vertex_count, properties


def _normalize_opacity(raw: np.ndarray) -> np.ndarray:
    if raw.size and (float(raw.min()) < 0.0 or float(raw.max()) > 1.0):
        return (1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))).astype(np.float32)
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def _decode_color(columns: dict[str, np.ndarray]) -> np.ndarray:
    if all(name in columns for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        raw = np.stack([columns["f_dc_0"], columns["f_dc_1"], columns["f_dc_2"]], axis=1).astype(np.float32)
        return np.clip(raw * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
    if all(name in columns for name in ("red", "green", "blue")):
        return np.stack([columns["red"], columns["green"], columns["blue"]], axis=1).astype(np.float32) / 255.0
    return np.full((len(next(iter(columns.values()))), 3), 0.8, dtype=np.float32)


_STRUCT_FORMAT = {
    "float": "f",
    "float32": "f",
    "double": "d",
    "uchar": "B",
    "uint8": "B",
    "char": "b",
    "int8": "b",
    "ushort": "H",
    "uint16": "H",
    "short": "h",
    "int16": "h",
    "uint": "I",
    "uint32": "I",
    "int": "i",
    "int32": "i",
}
