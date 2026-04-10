"""
OpenEnv multi-mode entry: tooling expects `server.app:app`.

Implementation lives in `mujoco_gym_env.server.app`.
"""

from mujoco_gym_env.server.app import app, main

__all__ = ["app", "main"]
