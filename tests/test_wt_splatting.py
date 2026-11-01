from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "plugins"))

from witwin_server.features.assets.binary.service import BinaryAssetService
from witwin_server.features.assets.binary.refs import POINT_CLOUD_FORMAT
from witwin_server.features.library.service import LibraryHandler


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


def test_pointcloud_component_is_owned_by_splatting_plugin() -> None:
    from wt_splatting.components.point_cloud import PointCloudsComponent

    component = PointCloudsComponent()
    values = component.to_dict()["values"]

    assert values["cloud"] is None
    assert values["point_count"] == 0
    assert values["colormap"] == "viridis"


def test_pointcloud_export_publishes_resource_ref_not_cloud_id() -> None:
    from wt_splatting.components.point_cloud import PointCloudsComponent

    component = PointCloudsComponent()
    component._owner = _Owner(object_id="pointcloud-object")
    component.set_points([[0, 0, 0], [1, 0, 0]], [0.25, 0.75])

    result = component.export_point_cloud()
    values = component.to_dict()["values"]

    assert result.startswith("Cached point cloud:")
    assert values["cloud"]["kind"] == "point_cloud"
    assert values["cloud"]["format"] == POINT_CLOUD_FORMAT
    assert "cloud_id" not in values
    assert _asset_service(component).get_bytes(values["cloud"]["assetId"]) is not None
    assert _component_update_fields(component) >= {"cloud"}


def test_pointcloud_prefab_contributed_by_splatting_plugin() -> None:
    from wt_splatting.library_items import _make_pointcloud_demo

    obj = _make_pointcloud_demo(_Context(name="Point Cloud"))
    component = obj.get_component("PointClouds")
    values = component.to_dict()["values"]

    assert obj.mesh_type == "Empty"
    assert values["point_count"] == 1000
    assert values["cloud"] is None


def test_pointcloud_library_prefab_exports_after_scene_attach() -> None:
    import asyncio
    import wt_splatting.library_items

    server = _LibraryServer()
    library = LibraryHandler(server)
    wt_splatting.library_items.register()

    result = asyncio.run(
        library._handle_create_from_library(
            {"item_id": "pointclouds", "position": [0, 0, 0]},
            websocket=None,
        )
    )

    obj = server.scene.get_object(result["object_id"])
    values = obj.get_component("PointClouds").to_dict()["values"]

    assert values["cloud"]["kind"] == "point_cloud"
    assert values["cloud"]["uri"].startswith("/scenes/default/assets/")


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
    def __init__(self, name: str = "Demo Splats") -> None:
        self.name = name
        self.position = [0.0, 0.0, 0.0]
        self.parent_id = None
        self.scene = None


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
    def __init__(self, object_id: str = "splat-object") -> None:
        self.id = object_id
        self.scene = _Scene()


class _LibraryServer:
    verbose = False
    main_scene_id = "default"

    def __init__(self) -> None:
        from witwin_server.core import Scene

        self.scene = Scene(scene_id=self.main_scene_id, role="main")
        self.scene._server = self
        self.handlers = {"binary_assets": _BinaryAssets()}

    def get_scene(self, scene_id: str | None = None):
        # One current scene, like the ProjectServer: any other id is not loaded.
        if not scene_id or scene_id == self.scene.scene_id:
            return self.scene
        return None

    async def broadcast(self, *args, **kwargs) -> None:
        pass

    def _safe_broadcast(self, *args, **kwargs) -> None:
        pass


def _asset_service(component):
    return component._owner.scene._server.handlers["binary_assets"].service


def _component_update_fields(component) -> set[str]:
    return {
        payload["field_name"]
        for event_type, payload in component._owner.scene.events
        if event_type == "component_updated"
    }
