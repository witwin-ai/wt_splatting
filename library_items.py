from __future__ import annotations

from witwin_server import Library, SceneObject

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


def _splat_object(name: str) -> SceneObject:
    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(GaussianSplatRendererComponent())
    return obj


register()
