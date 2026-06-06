from __future__ import annotations

import numpy as np

from witwin_server import Library, SceneObject

from .components.point_cloud import PointCloudsComponent
from .components.splat_renderer import GaussianSplatRendererComponent


_CATEGORY = "splatting"


def _make_3dgs_demo(ctx):
    obj = _splat_object(ctx.name)
    obj.get_component("GaussianSplatRenderer").set_field("splat_kind", "3dgs", force=True)
    return obj


def _make_2dgs_demo(ctx):
    obj = _splat_object(ctx.name)
    obj.get_component("GaussianSplatRenderer").set_field("splat_kind", "2dgs", force=True)
    return obj


def _make_pointcloud_demo(ctx):
    obj = SceneObject(name=ctx.name, mesh_type="Empty")
    component = PointCloudsComponent()
    obj.add_component(component)

    n_points = 1000
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    radius = np.random.uniform(0, 2, n_points) ** (1 / 3)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    positions = np.stack([x, y, z], axis=1)
    heat = np.linalg.norm(positions, axis=1) / 2.0
    extra_data = [
        {
            "id": i,
            "label": f"Point_{i}",
            "distance": float(np.linalg.norm(positions[i])),
            "sector": f"S{i % 8}",
        }
        for i in range(n_points)
    ]

    component.set_points(positions, heat, extra_data)
    component.set_field("colormap", "viridis", force=True)
    component.set_field("point_size", 0.06, force=True)
    component.set_field("opacity", 1.0, force=True)
    return obj


def register() -> None:
    Library.register_category(_CATEGORY, "Splatting", icon="sparkles", order=22)
    Library.register_item(
        "splatting_3dgs_demo",
        "3D Gaussian Splats",
        _CATEGORY,
        icon="sparkles",
        object_type="empty",
        factory=_make_3dgs_demo,
        description="3D Gaussian Splat renderer with an in-component random generator",
    )
    Library.register_item(
        "splatting_2dgs_demo",
        "2D Gaussian Splats",
        _CATEGORY,
        icon="scatter-chart",
        object_type="empty",
        factory=_make_2dgs_demo,
        description="2D Gaussian Splat renderer with an in-component random generator",
    )
    Library.register_item(
        "pointclouds",
        "Point Cloud",
        _CATEGORY,
        icon="scattered-points",
        object_type="pointcloud",
        factory=_make_pointcloud_demo,
        description="Point cloud renderer with generated point data",
    )


def _splat_object(name: str) -> SceneObject:
    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(GaussianSplatRendererComponent())
    return obj


register()
