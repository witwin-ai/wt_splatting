from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np


GAUSSIAN_SPLATS_KIND = "gaussian_splats"
GAUSSIAN_SPLATS_2D_KIND = "gaussian_splats_2d"
WITWIN_2DGS_QUAD_FORMAT = "witwin.2dgs.quad.v1"
W2GS_MAGIC = b"W2GS"
W2GS_VERSION = 1
_HEADER_SIZE = 24


def encode_2d_gaussian_splats(
    *,
    centers: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    sigma_uv: np.ndarray,
    opacity: np.ndarray,
    color_dc: np.ndarray,
    sh_degree: int = 0,
    flags: int = 0,
    attributes: dict[str, Any] | None = None,
) -> bytes:
    arrays = _validate_2d_arrays(
        centers=centers,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        sigma_uv=sigma_uv,
        opacity=opacity,
        color_dc=color_dc,
    )
    splat_count = int(arrays["centers"].shape[0])
    attrs = {"layout": "dense_float32", **(attributes or {})}
    attrs_bytes = json.dumps(attrs, separators=(",", ":")).encode("utf-8")
    padding = b"\x00" * (_align4(_HEADER_SIZE + len(attrs_bytes)) - (_HEADER_SIZE + len(attrs_bytes)))
    header = W2GS_MAGIC + struct.pack("<IIIII", W2GS_VERSION, splat_count, flags, int(sh_degree), len(attrs_bytes))
    body = b"".join(
        arrays[key].tobytes()
        for key in ("centers", "tangent_u", "tangent_v", "sigma_uv", "opacity", "color_dc")
    )
    return header + attrs_bytes + padding + body


def decode_2d_gaussian_splat_header(data: bytes) -> dict[str, Any]:
    if len(data) < _HEADER_SIZE:
        raise ValueError("2D Gaussian splat payload is too small")
    if data[:4] != W2GS_MAGIC:
        raise ValueError("Invalid 2D Gaussian splat payload: bad magic number")
    version, splat_count, flags, sh_degree, attrs_len = struct.unpack("<IIIII", data[4:_HEADER_SIZE])
    attrs_start = _HEADER_SIZE
    attrs_end = attrs_start + attrs_len
    if len(data) < attrs_end:
        raise ValueError("2D Gaussian splat payload is missing attributes JSON")
    attrs = json.loads(data[attrs_start:attrs_end].decode("utf-8")) if attrs_len else {}
    payload_offset = _align4(attrs_end)
    expected_bytes = payload_offset + splat_count * (3 + 3 + 3 + 2 + 1 + 3) * 4
    if len(data) < expected_bytes:
        raise ValueError("2D Gaussian splat payload is truncated")
    return {
        "magic": "W2GS",
        "version": version,
        "splat_count": splat_count,
        "flags": flags,
        "sh_degree": sh_degree,
        "attributes": attrs,
        "payload_offset": payload_offset,
        "format": WITWIN_2DGS_QUAD_FORMAT,
    }


def _align4(value: int) -> int:
    return (int(value) + 3) & ~3


def _validate_2d_arrays(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    centers = _float_array("centers", arrays["centers"], (-1, 3))
    splat_count = int(centers.shape[0])
    checked = {
        "centers": centers,
        "tangent_u": _float_array("tangent_u", arrays["tangent_u"], (splat_count, 3)),
        "tangent_v": _float_array("tangent_v", arrays["tangent_v"], (splat_count, 3)),
        "sigma_uv": _float_array("sigma_uv", arrays["sigma_uv"], (splat_count, 2)),
        "opacity": _float_array("opacity", arrays["opacity"], (splat_count,)),
        "color_dc": _float_array("color_dc", arrays["color_dc"], (splat_count, 3)),
    }
    return checked


def _float_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if len(shape) == 2 and shape[0] == -1:
        if array.ndim != 2 or array.shape[1] != shape[1]:
            raise ValueError(f"{name} must have shape Nx{shape[1]}, got {array.shape}")
        return array
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array
