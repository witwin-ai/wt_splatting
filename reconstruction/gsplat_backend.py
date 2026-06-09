from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .colmap import ColmapRunResult
from .config import GsplatTrainingConfig


@dataclass(frozen=True)
class GsplatAvailability:
    available: bool
    missing: list[str]
    message: str = ""


@dataclass(frozen=True)
class GsplatTrainResult:
    success: bool
    export_path: Path | None
    message: str


def training_payload(config: GsplatTrainingConfig) -> dict[str, int | str]:
    return {
        "device": config.device,
        "max_steps": config.max_steps,
        "sh_degree": config.sh_degree,
    }


class GsplatBackend:
    """Thin adapter around an installed gsplat training entrypoint."""

    def __init__(self, adapter: Callable[..., Path | str | None] | None = None) -> None:
        self._adapter = adapter

    def check_available(self) -> GsplatAvailability:
        missing = [name for name in ("torch", "gsplat") if importlib.util.find_spec(name) is None]
        if missing:
            return GsplatAvailability(False, missing, f"Missing dependencies: {', '.join(missing)}")
        return GsplatAvailability(True, [], "torch and gsplat are available")

    def train(
        self,
        config: GsplatTrainingConfig,
        colmap_result: ColmapRunResult,
        run_dir: Path,
        progress: Callable[[float, str], None] | None = None,
    ) -> GsplatTrainResult:
        availability = self.check_available()
        if not availability.available:
            return GsplatTrainResult(False, None, availability.message)

        run_dir = Path(run_dir)
        gsplat_dir = run_dir / "gsplat"
        export_dir = gsplat_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        payload = training_payload(config)
        (gsplat_dir / "training_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        try:
            export_path = self._invoke_adapter(config, colmap_result.dense_dir, export_dir, progress)
        except Exception as exc:
            return GsplatTrainResult(False, None, f"gsplat training failed: {exc}")

        discovered = Path(export_path) if export_path else _find_export(export_dir)
        if discovered is None:
            return GsplatTrainResult(False, None, f"gsplat export was not produced in {export_dir}")
        return GsplatTrainResult(True, discovered, "gsplat training complete")

    def _invoke_adapter(
        self,
        config: GsplatTrainingConfig,
        dense_dir: Path | None,
        export_dir: Path,
        progress: Callable[[float, str], None] | None,
    ) -> Path | str | None:
        if dense_dir is None:
            raise ValueError("COLMAP dense directory is required")
        if self._adapter is not None:
            return self._adapter(config=config, colmap_dir=dense_dir, export_dir=export_dir, progress=progress)

        adapter_spec = os.environ.get("WT_SPLATTING_GSPLAT_ADAPTER", "").strip()
        if adapter_spec:
            module_name, _, attr = adapter_spec.partition(":")
            module = importlib.import_module(module_name)
            adapter = getattr(module, attr or "train")
            return adapter(config=config, colmap_dir=dense_dir, export_dir=export_dir, progress=progress)

        script = os.environ.get("WT_SPLATTING_GSPLAT_TRAIN_SCRIPT", "").strip()
        if script:
            command = [
                sys.executable,
                script,
                "--data",
                str(dense_dir),
                "--output",
                str(export_dir),
                "--max-steps",
                str(config.max_steps),
                "--sh-degree",
                str(config.sh_degree),
                "--device",
                config.device,
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or "external trainer failed").strip())
            return _find_export(export_dir)

        raise RuntimeError(
            "No gsplat training adapter configured. Set WT_SPLATTING_GSPLAT_ADAPTER or WT_SPLATTING_GSPLAT_TRAIN_SCRIPT."
        )


def _find_export(export_dir: Path) -> Path | None:
    for suffix in (".ply", ".splat", ".spz", ".ksplat"):
        matches = sorted(Path(export_dir).glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None
