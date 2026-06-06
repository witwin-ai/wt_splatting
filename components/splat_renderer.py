from __future__ import annotations

import struct
from typing import Any
from uuid import uuid4

import numpy as np

from witwin_server.core.components.core import (
    Component,
    bool_field,
    button,
    component,
    float_field,
    int_field,
    string_field,
)
from witwin_server.features.assets.binary.service import BinaryAssetService

from ..io.formats import GAUSSIAN_SPLATS_KIND, WITWIN_2DGS_QUAD_FORMAT
from ..io.loaders import register_2d_splats_arrays

SPLAT_KINDS = ("3dgs", "2dgs")
SORT_MODES = ("gpu_depth", "gpu_weighted")
LOD_MODES = ("auto", "full", "off")
DEFAULT_RANDOM_SPLAT_COUNT = 512


@component(name="GaussianSplatRenderer", category="Rendering/Splatting")
class GaussianSplatRendererComponent(Component):
    """Component contract for 3DGS and 2DGS Gaussian Splatting resources."""

    splat_kind = string_field("3dgs", options=list(SPLAT_KINDS), group="Data", enum_toggle=True)
    format = string_field("", group="Data")
    splat_count = int_field(0, readonly=True, group="Data")

    sh_degree = int_field(0, min=0, max=3, group="Rendering")
    opacity_scale = float_field(1.0, min=0.0, max=5.0, step=0.01, group="Rendering")
    scale_modifier = float_field(1.0, min=0.0, max=5.0, step=0.01, group="Rendering")
    sort_mode = string_field("gpu_depth", options=list(SORT_MODES), group="Rendering", enum_toggle=True)

    max_splats = int_field(1_500_000, min=1, group="Budget")
    max_2d_splats = int_field(262_144, min=1, group="Budget")
    lod = string_field("auto", options=list(LOD_MODES), group="Budget", enum_toggle=True)
    debug_bounds = bool_field(False, group="Debug")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._values.setdefault("splats", None)
        self._sync_budget()

    def set_splats_ref(
        self,
        ref: dict[str, Any] | None,
        splat_count: int | None = None,
        splat_kind: str | None = None,
        format: str | None = None,
    ) -> None:
        if splat_kind is not None and splat_kind not in SPLAT_KINDS:
            raise ValueError(f"Unsupported splat_kind: {splat_kind}")
        self._values["splats"] = ref
        if splat_count is not None:
            self._values["splat_count"] = int(splat_count)
        if splat_kind is not None:
            self.set_field("splat_kind", splat_kind, silent=True, force=True)
        if format is not None:
            self.set_field("format", format, silent=True, force=True)
        self._publish_component_field("splats", ref)
        if splat_count is not None:
            self._publish_component_field("splat_count", int(splat_count))
        if splat_kind is not None:
            self._publish_component_field("splat_kind", splat_kind)
        if format is not None:
            self._publish_component_field("format", format)

    @button(display_name="Generate Random Splats", description="Generate random splats for the current splat kind")
    def generate_random_splats(
        self,
        params: dict[str, Any] | None = None,
        *,
        count: int | None = None,
        seed: int | None = None,
    ) -> str:
        params = params or {}
        splat_count = int(count if count is not None else params.get("count", DEFAULT_RANDOM_SPLAT_COUNT))
        splat_count = max(1, min(splat_count, int(self.max_splats)))
        seed_value = seed if seed is not None else params.get("seed")
        service = self._asset_service()

        if self.splat_kind == "2dgs":
            arrays = _random_2d_arrays(splat_count, seed_value)
            ref = register_2d_splats_arrays(
                service,
                **arrays,
                asset_id=f"random-2dgs-{uuid4().hex[:8]}.w2gs",
            )
            self.set_splats_ref(
                ref.to_dict(),
                splat_count=splat_count,
                splat_kind="2dgs",
                format=WITWIN_2DGS_QUAD_FORMAT,
            )
            return f"Generated {splat_count} random 2DGS splats: {ref.asset_id}"

        payload = _random_splat_payload(splat_count, seed_value)
        asset_id = f"random-3dgs-{uuid4().hex[:8]}.splat"
        ref = service.register_bytes(
            kind=GAUSSIAN_SPLATS_KIND,
            format="spark.splat",
            data=payload,
            shape=[splat_count],
            attributes={"splat_kind": "3dgs", "splat_count": splat_count},
            asset_id=asset_id,
        )
        self.set_splats_ref(
            ref.to_dict(),
            splat_count=splat_count,
            splat_kind="3dgs",
            format="spark.splat",
        )
        return f"Generated {splat_count} random 3DGS splats: {ref.asset_id}"

    def set_field(self, field_name: str, value: Any, *, silent: bool = False, force: bool = False):
        if field_name == "sort_mode" and value == "cpu":
            raise ValueError("GaussianSplatRenderer does not support CPU sorting")
        if field_name == "splat_kind" and value not in SPLAT_KINDS:
            raise ValueError(f"Unsupported splat_kind: {value}")
        if field_name == "sort_mode" and value not in SORT_MODES:
            raise ValueError(f"Unsupported sort_mode: {value}")
        if field_name == "lod" and value not in LOD_MODES:
            raise ValueError(f"Unsupported lod mode: {value}")
        result = super().set_field(field_name, value, silent=silent, force=force)
        if field_name in {"max_splats", "max_2d_splats", "lod"}:
            self._sync_budget()
        return result

    def _sync_budget(self) -> None:
        self._values["budget"] = {
            "max_splats": int(self.max_splats),
            "max_2d_splats": int(self.max_2d_splats),
            "lod": self.lod,
        }

    def _asset_service(self) -> BinaryAssetService:
        scene = getattr(getattr(self, "_owner", None), "scene", None)
        server = getattr(scene, "_server", None)
        handler = getattr(server, "handlers", {}).get("binary_assets") if server is not None else None
        service = getattr(handler, "service", None)
        if not isinstance(service, BinaryAssetService):
            raise RuntimeError("BinaryAssetHandler not available")
        return service

    def _publish_component_field(self, field_name: str, value: Any) -> None:
        owner = getattr(self, "_owner", None)
        scene = getattr(owner, "scene", None)
        if scene is not None and hasattr(scene, "_emit"):
            scene._emit(
                "component_updated",
                object_id=getattr(owner, "id", ""),
                component_name=self._component_name,
                field_name=field_name,
                value=value,
            )


def _random_splat_payload(count: int, seed: int | None = None) -> bytes:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 0.45, size=(count, 3)).astype(np.float32)
    centers[:, 1] += 0.6
    scales = rng.uniform(0.035, 0.11, size=(count, 3)).astype(np.float32)
    colors = rng.integers(48, 256, size=(count, 3), dtype=np.uint8)
    alpha = rng.integers(150, 235, size=(count, 1), dtype=np.uint8)
    rotations = _random_quaternion_bytes(rng, count)
    rgba = np.concatenate([colors, alpha], axis=1)
    return b"".join(
        struct.pack("<3f3f4B4B", *centers[i], *scales[i], *rgba[i], *rotations[i])
        for i in range(count)
    )


def _random_2d_arrays(count: int, seed: int | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 0.55, size=(count, 3)).astype(np.float32)
    centers[:, 2] *= 0.12
    centers[:, 1] += 0.45
    tangent_u, tangent_v = _random_tangent_frames(rng, count)
    sigma_uv = rng.uniform(0.025, 0.09, size=(count, 2)).astype(np.float32)
    opacity = rng.uniform(0.45, 0.9, size=count).astype(np.float32)
    color_dc = rng.uniform(0.15, 1.0, size=(count, 3)).astype(np.float32)
    return {
        "centers": centers,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "sigma_uv": sigma_uv,
        "opacity": opacity,
        "color_dc": color_dc,
    }


def _random_tangent_frames(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    normals = rng.normal(size=(count, 3)).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)

    reference = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (count, 1))
    near_vertical = np.abs(normals[:, 1]) > 0.92
    reference[near_vertical] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    tangent_u = np.cross(reference, normals).astype(np.float32)
    tangent_u /= np.maximum(np.linalg.norm(tangent_u, axis=1, keepdims=True), 1e-6)
    tangent_v = np.cross(normals, tangent_u).astype(np.float32)

    roll = rng.uniform(0.0, 2.0 * np.pi, size=(count, 1)).astype(np.float32)
    cos_roll = np.cos(roll).astype(np.float32)
    sin_roll = np.sin(roll).astype(np.float32)
    rolled_u = tangent_u * cos_roll + tangent_v * sin_roll
    rolled_v = -tangent_u * sin_roll + tangent_v * cos_roll
    return rolled_u.astype(np.float32), rolled_v.astype(np.float32)


def _random_quaternion_bytes(rng: np.random.Generator, count: int) -> np.ndarray:
    values = rng.normal(size=(count, 4)).astype(np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
    return np.clip(np.rint((values * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
