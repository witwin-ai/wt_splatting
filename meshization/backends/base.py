from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import MeshizationConfig


@dataclass(frozen=True)
class BackendMeshResult:
    mesh_path: Path
    vertex_count: int
    face_count: int
    method: str
    summary: str


class ExternalMeshBackend:
    method = "external"
    script_env = ""

    def run(
        self,
        config: MeshizationConfig,
        run_dir: Path,
        progress: Callable[[float, str], None] | None = None,
    ) -> BackendMeshResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(config, run_dir)
        if progress is not None:
            progress(0.35, f"run {self.method} backend")
        completed = subprocess.run(command, cwd=self._cwd(config), capture_output=True, text=True, check=False)
        log_path = run_dir / f"{self.method}.log"
        log_path.write_text((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or f"{self.method} backend failed").strip()
            raise RuntimeError(detail)
        mesh_path = run_dir / f"{config.output_name}.ply"
        if not mesh_path.is_file():
            mesh_path = self._find_mesh(run_dir)
        vertex_count, face_count = count_mesh_ply(mesh_path)
        return BackendMeshResult(
            mesh_path=mesh_path,
            vertex_count=vertex_count,
            face_count=face_count,
            method=self.method,
            summary=f"{self.method}: {vertex_count} vertices, {face_count} faces",
        )

    def build_command(self, config: MeshizationConfig, run_dir: Path) -> list[str]:
        script = self._script(config)
        python = config.backend_python or Path(sys.executable).resolve()
        mesh_path = run_dir / f"{config.output_name}.ply"
        command = [
            str(python),
            str(script),
            "--input",
            str(config.source_path),
            "--output-dir",
            str(run_dir),
            "--mesh-output",
            str(mesh_path),
            "--iterations",
            str(config.full_convergence_steps),
        ]
        if config.backend_repo is not None:
            command.extend(["--repo", str(config.backend_repo)])
        return command

    def _script(self, config: MeshizationConfig) -> Path:
        value = config.backend_script or _env_path(self.script_env)
        if value is None or not value.is_file():
            raise ValueError(f"backend_script is required for {self.method}; set the component field or {self.script_env}")
        return value

    def _cwd(self, config: MeshizationConfig) -> str | None:
        return str(config.backend_repo) if config.backend_repo is not None and config.backend_repo.is_dir() else None

    def _find_mesh(self, run_dir: Path) -> Path:
        matches = sorted(run_dir.rglob("*.ply"))
        if not matches:
            raise RuntimeError(f"{self.method} backend did not produce a .ply mesh under {run_dir}")
        return matches[0]


def count_mesh_ply(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    vertex_count = 0
    face_count = 0
    with path.open("rb") as fh:
        for raw_line in fh:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            elif line.startswith("element face "):
                face_count = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
    return vertex_count, face_count


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).resolve() if value else None
