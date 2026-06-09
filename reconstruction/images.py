from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ImageSequenceConfig


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageManifestEntry:
    original_path: Path
    staged_path: Path
    width: int | None = None
    height: int | None = None
    warning: str | None = None

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        return {
            "original_path": _display_path(self.original_path, project_root),
            "staged_path": str(self.staged_path),
            "width": self.width,
            "height": self.height,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class ImageManifest:
    image_dir: Path
    staged_dir: Path
    image_count: int
    images: list[ImageManifestEntry]
    warnings: list[str]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        return {
            "image_dir": _display_path(self.image_dir, project_root),
            "staged_dir": str(self.staged_dir),
            "image_count": self.image_count,
            "images": [entry.to_dict(project_root) for entry in self.images],
            "warnings": list(self.warnings),
        }


def resolve_project_relative_dir(project_root: Path, value: str | Path) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image directory must stay inside the Project: {resolved}") from exc
    if not resolved.is_dir():
        raise ValueError(f"Image directory does not exist: {resolved}")
    return resolved


def discover_images(image_dir: Path, max_images: int = 0) -> list[Path]:
    source = Path(image_dir)
    if not source.is_dir():
        raise ValueError(f"Image directory does not exist: {source}")
    images = sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=_natural_key,
    )
    if max_images > 0:
        return images[:max_images]
    return images


def build_image_manifest(project_root: Path, config: ImageSequenceConfig, run_dir: Path) -> ImageManifest:
    image_dir = resolve_project_relative_dir(project_root, config.image_dir)
    images = discover_images(image_dir, config.max_images)
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")

    staged_dir = Path(run_dir) / "images"
    staged_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ImageManifestEntry] = []
    warnings: list[str] = []
    for index, source in enumerate(images, start=1):
        staged = staged_dir / f"{index:06d}{source.suffix.lower()}"
        width, height, warning = _stage_image(source, staged, config.downscale_factor)
        if warning:
            warnings.append(f"{source.name}: {warning}")
        entries.append(
            ImageManifestEntry(
                original_path=source,
                staged_path=staged,
                width=width,
                height=height,
                warning=warning,
            )
        )

    manifest = ImageManifest(
        image_dir=image_dir,
        staged_dir=staged_dir,
        image_count=len(entries),
        images=entries,
        warnings=warnings,
    )
    manifest_path = Path(run_dir) / "input_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(Path(project_root).resolve()), indent=2), encoding="utf-8")
    return manifest


def _stage_image(source: Path, staged: Path, downscale_factor: int) -> tuple[int | None, int | None, str | None]:
    try:
        from PIL import Image
    except ImportError:
        shutil.copy2(source, staged)
        return None, None, "Pillow is unavailable; dimensions were not read"

    try:
        with Image.open(source) as image:
            image.load()
            if downscale_factor > 1:
                size = (
                    max(1, image.width // downscale_factor),
                    max(1, image.height // downscale_factor),
                )
                image = image.resize(size)
            image.save(staged)
            return int(image.width), int(image.height), None
    except Exception as exc:
        shutil.copy2(source, staged)
        return None, None, f"could not read dimensions: {exc}"


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _display_path(path: Path, project_root: Path | None) -> str:
    if project_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
