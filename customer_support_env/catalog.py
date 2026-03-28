from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_task_catalog() -> list[dict[str, Any]]:
    catalog_path = Path(__file__).resolve().parent / "tasks" / "catalog.json"
    return json.loads(catalog_path.read_text(encoding="utf-8"))
