from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .config import ColmapConfig
from .images import ImageManifest


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ColmapRunResult:
    success: bool
    sparse_dir: Path
    dense_dir: Path | None
    message: str
    commands: list[list[str]] = field(default_factory=list)
    registered_images: int = 0


def build_colmap_commands(config: ColmapConfig, manifest: ImageManifest, colmap_dir: Path) -> list[list[str]]:
    colmap_dir = Path(colmap_dir)
    database = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    dense = colmap_dir / "dense"
    matcher_command = "sequential_matcher" if config.matcher == "sequential" else "exhaustive_matcher"
    return [
        [
            config.colmap_binary,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(manifest.staged_dir),
            "--ImageReader.camera_model",
            config.camera_model,
        ],
        [
            config.colmap_binary,
            matcher_command,
            "--database_path",
            str(database),
        ],
        [
            config.colmap_binary,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(manifest.staged_dir),
            "--output_path",
            str(sparse),
        ],
        [
            config.colmap_binary,
            "image_undistorter",
            "--image_path",
            str(manifest.staged_dir),
            "--input_path",
            str(sparse / "0"),
            "--output_path",
            str(dense),
            "--output_type",
            "COLMAP",
        ],
    ]


class ColmapRunner:
    def __init__(self, run_process: ProcessRunner | None = None) -> None:
        self._run_process = run_process or subprocess.run

    def run(self, config: ColmapConfig, manifest: ImageManifest, run_dir: Path) -> ColmapRunResult:
        run_dir = Path(run_dir)
        colmap_dir = run_dir / "colmap"
        sparse_dir = colmap_dir / "sparse"
        dense_dir = colmap_dir / "dense"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)

        if manifest.image_count < 2:
            return ColmapRunResult(False, sparse_dir, None, "too few registered images", registered_images=manifest.image_count)

        commands = build_colmap_commands(config, manifest, colmap_dir)
        for command in commands:
            try:
                completed = self._run_process(
                    command,
                    cwd=str(colmap_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return ColmapRunResult(False, sparse_dir, None, f"missing executable: {config.colmap_binary}", commands=commands)

            if int(getattr(completed, "returncode", 1)) != 0:
                stderr = str(getattr(completed, "stderr", "") or "").strip()
                stdout = str(getattr(completed, "stdout", "") or "").strip()
                detail = stderr or stdout or "no output"
                return ColmapRunResult(
                    False,
                    sparse_dir,
                    None,
                    f"COLMAP exited with code {completed.returncode}: {detail}",
                    commands=commands,
                )

        registered_images = _registered_image_count(sparse_dir)
        if registered_images < 2:
            return ColmapRunResult(
                False,
                sparse_dir,
                None,
                "missing sparse model or too few registered images",
                commands=commands,
                registered_images=registered_images,
            )
        return ColmapRunResult(True, sparse_dir, dense_dir, "COLMAP model built", commands=commands, registered_images=registered_images)


def _registered_image_count(sparse_dir: Path) -> int:
    for model_dir in [sparse_dir / "0", sparse_dir]:
        images_bin = model_dir / "images.bin"
        images_txt = model_dir / "images.txt"
        if images_bin.exists():
            return 2
        if images_txt.exists():
            return _count_images_txt(images_txt)
    return 0


def _count_images_txt(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return max(0, count // 2)
