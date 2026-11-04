"""``splat_asset_create_object`` turns a ``.wtsplat`` manifest into a GaussianSplatRenderer object.

The op handler moved here from Studio's kernel (``features/assets/splats``,
2026-11-03, D16 review B-SPLAT), together with this test.
"""
from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

from witwin_server.api import BinaryAssetService, Scene

from wt_splatting.assets.placement import create_splat_object


def test_splat_asset_op_creates_gaussian_splat_renderer_object(tmp_path: Path) -> None:
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

    result = asyncio.run(create_splat_object(
        server, {"path": "assets/splatting/kitchen.wtsplat", "position": [1.0, 2.0, 3.0]}))

    obj = server.scene.get_object(result["object_id"])
    renderer = obj.get_component("GaussianSplatRenderer")
    values = renderer.to_dict()["values"]
    assert result["type"] == "splat_asset_create_object_result"
    assert values["splats"]["assetId"].endswith(".ply")
    assert values["splats"]["format"] == "3dgs.ply"
    assert values["splat_kind"] == "3dgs"
    assert values["splat_count"] == 2
    assert server.handlers["binary_assets"].service.get_bytes(values["splats"]["assetId"]) == source.read_bytes()


def test_splat_asset_op_refuses_paths_outside_the_project(tmp_path: Path) -> None:
    server = _Server(tmp_path / "project")
    result = asyncio.run(create_splat_object(server, {"path": "../outside.wtsplat"}))
    assert result == {"type": "error", "message": "Path is outside the Project"}


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


class _BinaryAssets:
    def __init__(self) -> None:
        self.service = BinaryAssetService()


class _Server:
    """The parts of a ProjectServer the op reads: project folder, current Scene, binary assets."""

    verbose = False
    main_scene_id = "default"

    def __init__(self, project: Path) -> None:
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
