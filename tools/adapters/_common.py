from __future__ import annotations

import os
import shutil
import subprocess
import importlib.util
from pathlib import Path


MESH_SUFFIXES = (".ply", ".obj")


def require_repo_file(repo: str | Path, name: str) -> Path:
    root = Path(repo).resolve()
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Expected {name} under backend repo: {root}")
    return path


def run_logged(command: list[str], *, cwd: str | Path, log_path: str | Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), env=_backend_env(command), capture_output=True, text=True, check=False)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"command failed: {command[0]}").strip()
        raise RuntimeError(detail)


def find_newest_mesh(root: str | Path, *, newer_than: float | None = None) -> Path:
    base = Path(root).resolve()
    matches = [path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in MESH_SUFFIXES]
    if newer_than is not None:
        matches = [path for path in matches if path.stat().st_mtime >= newer_than]
    if not matches:
        raise FileNotFoundError(f"No mesh file found under {base}")
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def copy_or_convert_mesh(source: str | Path, target: str | Path) -> Path:
    src = Path(source).resolve()
    dst = Path(target).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".ply":
        if src != dst:
            shutil.copy2(src, dst)
        return dst
    if src.suffix.lower() == ".obj":
        try:
            import trimesh
        except Exception as exc:
            raise RuntimeError("trimesh is required to convert SuGaR OBJ output to PLY") from exc
        mesh = trimesh.load(str(src), force="mesh")
        mesh.export(str(dst))
        return dst
    raise ValueError(f"Unsupported mesh output suffix: {src.suffix}")


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _backend_env(command: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    if not command:
        return env
    executable = Path(command[0])
    if executable.name.lower().startswith("python") and executable.parent.exists():
        paths = [executable.parent / "Library" / "bin", executable.parent]
        existing = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([str(path) for path in paths] + ([existing] if existing else []))
    return env
