from __future__ import annotations

import numpy as np

from .config import MeshizationConfig
from .ply import GaussianSplatData


def extract_density_mesh(splats: GaussianSplatData, config: MeshizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from skimage.measure import marching_cubes
    except Exception as exc:
        raise RuntimeError("Meshization requires scikit-image. Install it in the runtime environment with: pip install scikit-image") from exc

    filtered = _filter_splats(splats, config)
    if filtered.count < 4:
        raise ValueError("Not enough visible Gaussians for mesh extraction")

    density, origin, spacing = _density_grid(filtered, config)
    if float(density.max()) <= config.iso_level:
        raise ValueError(f"Density field never reaches iso_level {config.iso_level}; max density is {float(density.max()):0.4f}")
    verts_ijk, faces, _, _ = marching_cubes(density, level=config.iso_level, spacing=(spacing[0], spacing[1], spacing[2]))
    vertices = verts_ijk.astype(np.float32) + origin.astype(np.float32)
    faces = faces.astype(np.uint32)
    colors = _nearest_colors(vertices, filtered.centers, filtered.colors)
    return vertices, faces, colors


def _filter_splats(splats: GaussianSplatData, config: MeshizationConfig) -> GaussianSplatData:
    mask = splats.opacities >= config.opacity_threshold
    centers = splats.centers[mask]
    scales = splats.scales[mask]
    opacities = splats.opacities[mask]
    colors = splats.colors[mask]
    if len(centers) > config.max_gaussians:
        order = np.argsort(opacities)[-config.max_gaussians :]
        centers = centers[order]
        scales = scales[order]
        opacities = opacities[order]
        colors = colors[order]
    return GaussianSplatData(centers=centers, scales=scales, opacities=opacities, colors=colors)


def _density_grid(splats: GaussianSplatData, config: MeshizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = np.maximum(np.mean(splats.scales, axis=1), 1.0e-4)
    expansion = np.max(radii) * config.sigma_radius
    bounds_min = splats.centers.min(axis=0) - expansion
    bounds_max = splats.centers.max(axis=0) + expansion
    extent = np.maximum(bounds_max - bounds_min, 1.0e-3)
    resolution = int(config.grid_resolution)
    spacing = extent / float(resolution - 1)
    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    for center, radius, opacity in zip(splats.centers, radii, splats.opacities):
        sigma = max(float(radius), float(np.max(spacing)) * 0.5)
        voxel_radius = np.ceil(config.sigma_radius * sigma / spacing).astype(int)
        center_idx = np.rint((center - bounds_min) / spacing).astype(int)
        lo = np.maximum(center_idx - voxel_radius, 0)
        hi = np.minimum(center_idx + voxel_radius + 1, resolution)
        if np.any(hi <= lo):
            continue
        xs = bounds_min[0] + np.arange(lo[0], hi[0], dtype=np.float32) * spacing[0]
        ys = bounds_min[1] + np.arange(lo[1], hi[1], dtype=np.float32) * spacing[1]
        zs = bounds_min[2] + np.arange(lo[2], hi[2], dtype=np.float32) * spacing[2]
        dx2 = ((xs - center[0]) / sigma) ** 2
        dy2 = ((ys - center[1]) / sigma) ** 2
        dz2 = ((zs - center[2]) / sigma) ** 2
        local = np.exp(-0.5 * (dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :])).astype(np.float32)
        density[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] += float(opacity) * local
    return density, bounds_min.astype(np.float32), spacing.astype(np.float32)


def _nearest_colors(vertices: np.ndarray, centers: np.ndarray, colors: np.ndarray) -> np.ndarray:
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    result = np.empty((len(vertices), 3), dtype=np.float32)
    for start in range(0, len(vertices), 4096):
        chunk = vertices[start : start + 4096]
        distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        result[start : start + len(chunk)] = colors[np.argmin(distances, axis=1)]
    return result
