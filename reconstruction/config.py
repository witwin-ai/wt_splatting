from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VALID_DOWNSCALE_FACTORS = {1, 2, 4, 8}
VALID_MATCHERS = {"sequential", "exhaustive"}
VALID_EXPORT_FORMATS = {"ply"}
VALID_DEVICES = {"cuda", "cpu"}


@dataclass(frozen=True)
class ImageSequenceConfig:
    image_dir: str
    capture_id: str = "capture"
    downscale_factor: int = 1
    max_images: int = 0

    @classmethod
    def from_component_values(cls, values: dict[str, Any]) -> "ImageSequenceConfig":
        image_dir = str(values.get("image_dir") or "").strip()
        if not image_dir:
            raise ValueError("image_dir is required")
        if Path(image_dir).is_absolute():
            raise ValueError("image_dir must be Project-relative")
        downscale_factor = _int_value(values.get("downscale_factor", 1), "downscale_factor")
        if downscale_factor not in VALID_DOWNSCALE_FACTORS:
            raise ValueError("downscale_factor must be one of 1, 2, 4, or 8")
        max_images = _int_value(values.get("max_images", 0), "max_images")
        if max_images < 0:
            raise ValueError("max_images must be >= 0")
        return cls(
            image_dir=image_dir.replace("\\", "/"),
            capture_id=_safe_token(values.get("capture_id") or "capture", "capture_id"),
            downscale_factor=downscale_factor,
            max_images=max_images,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColmapConfig:
    colmap_binary: str = "colmap"
    camera_model: str = "OPENCV"
    matcher: str = "sequential"

    @classmethod
    def from_component_values(cls, values: dict[str, Any]) -> "ColmapConfig":
        matcher = str(values.get("matcher") or "sequential")
        if matcher not in VALID_MATCHERS:
            raise ValueError("matcher must be sequential or exhaustive")
        colmap_binary = str(values.get("colmap_binary") or "colmap").strip()
        if not colmap_binary:
            raise ValueError("colmap_binary is required")
        camera_model = str(values.get("camera_model") or "OPENCV").strip()
        if not camera_model:
            raise ValueError("camera_model is required")
        return cls(colmap_binary=colmap_binary, camera_model=camera_model, matcher=matcher)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GsplatTrainingConfig:
    device: str = "cuda"
    max_steps: int = 7000
    sh_degree: int = 3

    @classmethod
    def from_component_values(cls, values: dict[str, Any]) -> "GsplatTrainingConfig":
        device = str(values.get("device") or "cuda")
        if device not in VALID_DEVICES:
            raise ValueError("device must be cuda or cpu")
        max_steps = _int_value(values.get("max_steps", 7000), "max_steps")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        sh_degree = _int_value(values.get("sh_degree", 3), "sh_degree")
        if sh_degree < 0 or sh_degree > 3:
            raise ValueError("sh_degree must be between 0 and 3")
        return cls(device=device, max_steps=max_steps, sh_degree=sh_degree)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExportConfig:
    output_name: str = "reconstruction"
    export_format: str = "ply"

    @classmethod
    def from_component_values(cls, values: dict[str, Any]) -> "ExportConfig":
        export_format = str(values.get("export_format") or "ply")
        if export_format not in VALID_EXPORT_FORMATS:
            raise ValueError("export_format must be ply")
        return cls(
            output_name=_safe_token(values.get("output_name") or "reconstruction", "output_name"),
            export_format=export_format,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionConfig:
    image: ImageSequenceConfig
    colmap: ColmapConfig
    training: GsplatTrainingConfig
    export: ExportConfig

    @classmethod
    def from_component_values(cls, values: dict[str, Any]) -> "ReconstructionConfig":
        return cls(
            image=ImageSequenceConfig.from_component_values(values),
            colmap=ColmapConfig.from_component_values(values),
            training=GsplatTrainingConfig.from_component_values(values),
            export=ExportConfig.from_component_values(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image.to_dict(),
            "colmap": self.colmap.to_dict(),
            "training": self.training.to_dict(),
            "export": self.export.to_dict(),
        }


@dataclass(frozen=True)
class ReconstructionResult:
    run_id: str
    splat_path: Path
    splat_count: int = 0
    format: str = "spark.ply"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "splat_path": str(self.splat_path),
            "splat_count": self.splat_count,
            "format": self.format,
            "summary": self.summary,
        }


def _int_value(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _safe_token(value: Any, name: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{name} is required")
    if any(part in token for part in ("/", "\\", "..")):
        raise ValueError(f"{name} must be a simple name")
    return token
