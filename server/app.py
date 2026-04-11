"""
OpenEnv multi-mode entry: `server.app:app` and local `main()`.

Implementation: drug_discovery_env.server.app
"""

from __future__ import annotations

import os

from drug_discovery_env.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
