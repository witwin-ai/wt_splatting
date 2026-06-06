from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "plugins"))

from witwin_server.features.assets.binary.service import BinaryAssetService


def test_component_defaults() -> None:
    from wt_splatting.components.splat_renderer import GaussianSplatRendererComponent

    component = GaussianSplatRendererComponent()
    payload = component.to_dict()
    values = payload["values"]

    assert values["splats"] is None
    assert values["splat_kind"] == "3dgs"
    assert values["format"] == ""
    assert values["splat_count"] == 0
    assert values["sh_degree"] == 0
    assert values["opacity_scale"] == 1.0
    assert values["scale_modifier"] == 1.0
    assert values["sort_mode"] == "gpu_depth"
    assert values["max_splats"] == 1_500_000
    assert values["max_2d_splats"] == 262_144
    assert values["lod"] == "auto"
    assert values["debug_bounds"] is False
    assert values["budget"] == {"max_splats": 1_500_000, "max_2d_splats": 262_144, "lod": "auto"}
    assert "generate_random_splats" in {button["name"] for button in payload["buttons"]}


def test_component_rejects_cpu_sort_mode() -> None:
    from wt_splatting.components.splat_renderer import GaussianSplatRendererComponent

    component = GaussianSplatRendererComponent()

    with pytest.raises(ValueError):
        component.set_field("sort_mode", "cpu")


def test_set_splats_ref_sets_public_contract() -> None:
    from wt_splatting.components.splat_renderer import GaussianSplatRendererComponent

    ref = {
        "assetId": "demo.ply",
        "uri": "/scenes/default/assets/demo.ply",
        "kind": "gaussian_splats",
        "format": "spark.ply",
        "byteLength": 128,
        "contentHash": "sha256:abc",
    }
    component = GaussianSplatRendererComponent()
    component.set_splats_ref(ref, splat_count=7, splat_kind="3dgs", format="spark.ply")

    values = component.to_dict()["values"]
    assert values["splats"] == ref
    assert values["splat_count"] == 7
    assert values["splat_kind"] == "3dgs"
    assert values["format"] == "spark.ply"


def test_2dgs_header_round_trips() -> None:
    from wt_splatting.io.formats import decode_2d_gaussian_splat_header, encode_2d_gaussian_splats

    payload = encode_2d_gaussian_splats(
        centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        tangent_u=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        tangent_v=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        sigma_uv=np.array([[0.25, 0.5]], dtype=np.float32),
        opacity=np.array([0.75], dtype=np.float32),
        color_dc=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        sh_degree=1,
        attributes={"source": "test"},
    )

    header = decode_2d_gaussian_splat_header(payload)

    assert header["magic"] == "W2GS"
    assert header["version"] == 1
    assert header["splat_count"] == 1
    assert header["sh_degree"] == 1
    assert header["format"] == "witwin.2dgs.quad.v1"
    assert header["attributes"]["source"] == "test"
    assert header["payload_offset"] % 4 == 0


def test_2dgs_encoder_pads_attributes_to_float32_boundary() -> None:
    from wt_splatting.io.formats import decode_2d_gaussian_splat_header, encode_2d_gaussian_splats

    payload = encode_2d_gaussian_splats(
        centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        tangent_u=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        tangent_v=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        sigma_uv=np.array([[0.25, 0.5]], dtype=np.float32),
        opacity=np.array([0.75], dtype=np.float32),
        color_dc=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        attributes={"label": "abc"},
    )

    header = decode_2d_gaussian_splat_header(payload)

    assert header["payload_offset"] % 4 == 0
    assert len(payload) == header["payload_offset"] + (3 + 3 + 3 + 2 + 1 + 3) * 4


def test_loader_rejects_outside_project_path(tmp_path: Path) -> None:
    from wt_splatting.io.loaders import register_3d_splats_file

    project = tmp_path / "project"
    outside = tmp_path / "outside.ply"
    project.mkdir()
    outside.write_bytes(_tiny_ply())

    with pytest.raises(ValueError):
        register_3d_splats_file(BinaryAssetService(), outside, project_root=project)


def test_loader_requires_project_root(tmp_path: Path) -> None:
    from wt_splatting.io.loaders import register_3d_splats_file

    source = tmp_path / "demo.ply"
    source.write_bytes(_tiny_ply())

    with pytest.raises(ValueError):
        register_3d_splats_file(BinaryAssetService(), source)


def test_3dgs_file_ref_has_content_hash(tmp_path: Path) -> None:
    from wt_splatting.io.loaders import register_3d_splats_file

    source = tmp_path / "demo.ply"
    source.write_bytes(_tiny_ply())

    ref = register_3d_splats_file(BinaryAssetService(), source, project_root=tmp_path)

    payload = ref.to_dict()
    assert payload["kind"] == "gaussian_splats"
    assert payload["format"] == "spark.ply"
    assert payload["contentHash"].startswith("sha256:")
    assert payload["attributes"]["splat_kind"] == "3dgs"


def test_3dgs_asset_id_preserves_spark_extension(tmp_path: Path) -> None:
    from wt_splatting.io.loaders import register_3d_splats_file

    source = tmp_path / "demo-scene.ply"
    source.write_bytes(_tiny_ply())

    ref = register_3d_splats_file(BinaryAssetService(), source, project_root=tmp_path)

    assert ref.asset_id.endswith(".ply")
    assert ref.uri.endswith(".ply")


def test_3dgs_prefab_starts_empty_with_generate_button() -> None:
    from wt_splatting.library_items import _make_3dgs_demo

    obj = _make_3dgs_demo(_Context())
    values = obj.get_component("GaussianSplatRenderer").to_dict()["values"]

    assert obj.mesh_type == "Empty"
    assert values["splats"] is None
    assert values["splat_kind"] == "3dgs"
    assert values["splat_count"] == 0


def test_component_generate_random_3dgs_sets_splat_ref() -> None:
    from wt_splatting.components.splat_renderer import GaussianSplatRendererComponent

    component = GaussianSplatRendererComponent()
    component._owner = _Owner()

    result = component.click_button("generate_random_splats", {"count": 16, "seed": 7})
    values = component.to_dict()["values"]

    assert "Generated 16 random 3DGS splats" in result
    assert values["splats"]["format"] == "spark.splat"
    assert values["splats"]["assetId"].endswith(".splat")
    assert values["splat_kind"] == "3dgs"
    assert values["splat_count"] == 16
    assert _asset_service(component).get_bytes(values["splats"]["assetId"]) is not None
    assert _component_update_fields(component) >= {"splats", "splat_count", "splat_kind", "format"}


def test_2dgs_prefab_starts_empty_with_generate_button() -> None:
    from wt_splatting.library_items import _make_2dgs_demo

    obj = _make_2dgs_demo(_Context())
    values = obj.get_component("GaussianSplatRenderer").to_dict()["values"]

    assert obj.mesh_type == "Empty"
    assert values["splats"] is None
    assert values["splat_kind"] == "2dgs"
    assert values["splat_count"] == 0


def test_component_generate_random_2dgs_sets_aligned_ref() -> None:
    from wt_splatting.components.splat_renderer import GaussianSplatRendererComponent
    from wt_splatting.io.formats import decode_2d_gaussian_splat_header

    component = GaussianSplatRendererComponent()
    component._owner = _Owner()
    component.set_field("splat_kind", "2dgs", force=True)

    result = component.click_button("generate_random_splats", {"count": 16, "seed": 11})
    values = component.to_dict()["values"]
    data = _asset_service(component).get_bytes(values["splats"]["assetId"])

    assert "Generated 16 random 2DGS splats" in result
    assert values["splats"]["format"] == "witwin.2dgs.quad.v1"
    assert values["splat_kind"] == "2dgs"
    assert values["splat_count"] == 16
    assert data is not None
    assert decode_2d_gaussian_splat_header(data)["payload_offset"] % 4 == 0
    assert _component_update_fields(component) >= {"splats", "splat_count", "splat_kind", "format"}


def test_random_2dgs_uses_varied_orthonormal_tangent_frames() -> None:
    from wt_splatting.components.splat_renderer import _random_2d_arrays

    arrays = _random_2d_arrays(64, seed=21)
    tangent_u = arrays["tangent_u"]
    tangent_v = arrays["tangent_v"]
    normals = np.cross(tangent_u, tangent_v)
    rounded_normals = np.unique(np.round(normals, 3), axis=0)

    assert np.allclose(np.linalg.norm(tangent_u, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(tangent_v, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.einsum("ij,ij->i", tangent_u, tangent_v), 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    assert rounded_normals.shape[0] > 8


def _tiny_ply() -> bytes:
    return (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"end_header\n"
        b"0 0 0 255 0 0\n"
    )


class _Context:
    name = "Demo Splats"
    position = [0.0, 0.0, 0.0]
    parent_id = None
    scene = None


class _BinaryAssets:
    def __init__(self) -> None:
        self.service = BinaryAssetService()


class _Server:
    def __init__(self) -> None:
        self.handlers = {"binary_assets": _BinaryAssets()}


class _Scene:
    def __init__(self) -> None:
        self._server = _Server()
        self.events = []

    def _emit(self, event_type, **payload) -> None:
        self.events.append((event_type, payload))


class _Owner:
    id = "splat-object"

    def __init__(self) -> None:
        self.scene = _Scene()


def _asset_service(component):
    return component._owner.scene._server.handlers["binary_assets"].service


def _component_update_fields(component) -> set[str]:
    return {
        payload["field_name"]
        for event_type, payload in component._owner.scene.events
        if event_type == "component_updated"
    }
