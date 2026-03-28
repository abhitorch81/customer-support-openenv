from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_local_deps() -> None:
    project_root = Path(__file__).resolve().parents[1]
    deps_dir = project_root / ".deps"
    if deps_dir.exists():
        deps_path = str(deps_dir)
        if deps_path not in sys.path:
            # Keep the project-local fallback available without shadowing
            # packages already installed in the active virtual environment.
            sys.path.append(deps_path)
