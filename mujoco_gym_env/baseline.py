from __future__ import annotations

import numpy as np

from .environment import MuJoCoGymEnvironment
from .models import MuJoCoAction, MuJoCoObservation


class RandomPolicy:
    """Uniform random actions in [-1, 1] (matches common MuJoCo Box bounds)."""

    name = "random"

    def act(self, observation: MuJoCoObservation) -> MuJoCoAction:
        dim = max(1, observation.action_dim)
        ctrl = np.random.uniform(-1.0, 1.0, size=(dim,)).astype(np.float64).tolist()
        return MuJoCoAction(control=ctrl)


def main() -> None:
    """Run one random-policy episode per registered task (CLI / smoke)."""
    env = MuJoCoGymEnvironment()
    policy = RandomPolicy()
    try:
        for task in env.list_tasks():
            tid = task.metadata.task_id
            env.select_task(tid)
            obs = env.reset()
            total = 0.0
            while not obs.done:
                obs = env.step(policy.act(obs))
                total += float(obs.reward or 0.0)
            g = env.grade_current_episode()
            print(f"task={tid} return={total:.4f} score={g.score:.4f} passed={g.passed}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
