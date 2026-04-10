"""Smoke test: one MuJoCo episode with random actions (requires pip install ".[mujoco]")."""

from __future__ import annotations

import sys


def main() -> None:
    import numpy as np

    from mujoco_gym_env import MuJoCoGymEnvironment
    from mujoco_gym_env.models import MuJoCoAction

    env = MuJoCoGymEnvironment()
    obs = env.reset()
    total = 0.0
    for _ in range(30):
        a = np.random.uniform(-1.0, 1.0, size=(obs.action_dim,)).tolist()
        obs = env.step(MuJoCoAction(control=a))
        total += float(obs.reward or 0.0)
        if obs.done:
            break
    g = env.grade_current_episode()
    env.close()
    print(f"smoke_mujoco_gym ok steps={obs.step_count} return={total:.4f} score={g.score:.4f} passed={g.passed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_mujoco_gym failed: {exc}", file=sys.stderr)
        sys.exit(1)
