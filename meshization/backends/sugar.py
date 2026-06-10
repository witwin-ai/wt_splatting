from __future__ import annotations

import os
from pathlib import Path

from .base import ExternalMeshBackend


class SuGaRBackend(ExternalMeshBackend):
    method = "sugar"
    script_env = "WT_SPLATTING_SUGAR_SCRIPT"

    def _script(self, config):
        if config.backend_script is not None or os.environ.get(self.script_env, "").strip():
            return super()._script(config)
        return Path(__file__).resolve().parents[2] / "tools" / "adapters" / "sugar_official.py"
