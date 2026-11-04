"""WiTwin Gaussian Splatting plugin."""

from witwin_server.api import PluginContext

from . import components  # noqa: F401
from . import library_items  # noqa: F401
from .assets import placement


def activate(ctx: PluginContext) -> None:
    # Placing a .wtsplat file in the Scene is this Plugin's scene op.
    placement.register(ctx)
