from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_task_catalog() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parent / "tasks" / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))
