from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "requires_colmap: requires a configured COLMAP reconstruction capture")
    config.addinivalue_line("markers", "requires_gsplat: requires torch and gsplat in the active Python environment")
    config.addinivalue_line("markers", "requires_surface_backends: requires configured SuGaR and 2DGS backends")
