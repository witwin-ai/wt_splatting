from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "plugins"))

from witwin_server.features.assets.binary.service import BinaryAssetService


def test_classify_ply_distinguishes_3dgs_from_mesh(tmp_path: Path) -> None:
    from wt_splatting.assets.splat_manifest import classify_ply

    splat_path = tmp_path / "kitchen.ply"
    splat_path.write_bytes(_gaussian_ply_bytes())
    mesh_path = tmp_path / "proxy.ply"
    mesh_path.write_bytes(_mesh_ply_bytes())

    splat = classify_ply(splat_path)
    mesh = classify_ply(mesh_path)

    assert splat.kind == "gaussian_splat_3dgs"
    assert splat.splat_count == 2
    assert mesh.kind == "mesh_ply"


def test_wtsplat_manifest_stays_inside_project(tmp_path: Path) -> None:
    from wt_splatting.assets.splat_manifest import read_wtsplat_manifest, write_wtsplat_manifest

    project = tmp_path / "project"
    source = project / "assets" / "splatting" / "room.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_gaussian_ply_bytes())
    manifest = project / "assets" / "splatting" / "room.wtsplat"

    write_wtsplat_manifest(manifest, source, project_root=project)
    loaded = read_wtsplat_manifest(manifest, project_root=project)

    assert loaded.source_path == source.resolve()
    assert loaded.source_rel == "assets/splatting/room.ply"
    assert loaded.format == "3dgs.ply"
    assert loaded.splat_count == 2

    manifest.write_text(json.dumps({"source": "../outside.ply", "format": "3dgs.ply"}), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the Project"):
        read_wtsplat_manifest(manifest, project_root=project)


def test_splat_asset_handler_creates_gaussian_splat_renderer_object(tmp_path: Path) -> None:
    import wt_splatting  # noqa: F401
    from witwin_server.features.assets.splats.handlers import SplatAssetHandler

    project = tmp_path / "project"
    source = project / "assets" / "splatting" / "kitchen.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_gaussian_ply_bytes())
    manifest = project / "assets" / "splatting" / "kitchen.wtsplat"
    manifest.write_text(
        json.dumps({"source": "kitchen.ply", "format": "3dgs.ply", "splat_count": 2}),
        encoding="utf-8",
    )
    server = _Server(project)
    handler = SplatAssetHandler(server)

    result = asyncio.run(
        handler._create_object(
            {"path": "assets/splatting/kitchen.wtsplat", "position": [1.0, 2.0, 3.0]},
            websocket=None,
        )
    )

    obj = server.scene.get_object(result["object_id"])
    renderer = obj.get_component("GaussianSplatRenderer")
    values = renderer.to_dict()["values"]
    assert result["type"] == "splat_asset_create_object_result"
    assert values["splats"]["assetId"].endswith(".ply")
    assert values["splats"]["format"] == "3dgs.ply"
    assert values["splat_kind"] == "3dgs"
    assert values["splat_count"] == 2
    assert server.handlers["binary_assets"].service.get_bytes(values["splats"]["assetId"]) == source.read_bytes()


def _gaussian_ply_bytes() -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")
    rows = [
        (0.0, 0.0, 0.0, 0.8, 0.1, 0.2, 0.8, -2.3, -2.3, -2.3, 1.0, 0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0, 0.2, 0.7, 0.1, 0.6, -2.3, -2.3, -2.3, 1.0, 0.0, 0.0, 0.0),
    ]
    return header + b"".join(struct.pack("<14f", *row) for row in rows)


def _mesh_ply_bytes() -> bytes:
    return (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 3\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"element face 1\n"
        b"property list uchar int vertex_indices\n"
        b"end_header\n"
        b"0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )


class _BinaryAssets:
    def __init__(self) -> None:
        self.service = BinaryAssetService()


class _Server:
    verbose = False
    main_scene_id = "default"

    def __init__(self, project: Path) -> None:
        from witwin_server.core import Scene

        self.default_scene_dir = project
        self.scene = Scene(scene_id=self.main_scene_id, role="main")
        self.scene._server = self
        self.handlers = {"binary_assets": _BinaryAssets()}

    def get_scene(self, scene_id: str | None = None):
        return self.scene

    async def broadcast(self, *args, **kwargs) -> None:
        pass

    def _safe_broadcast(self, *args, **kwargs) -> None:
        pass
