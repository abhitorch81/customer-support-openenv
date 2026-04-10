from __future__ import annotations

from typing import Any
from uuid import uuid4

from .._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..models import (
    Difficulty,
    GraderResult,
    MuJoCoAction,
    MuJoCoObservation,
    MuJoCoState,
    TaskDescriptor,
    TaskMetadata,
)


def _strict_unit_interval(value: float) -> float:
    eps = 1e-4
    if value <= 0.0:
        return eps
    if value >= 1.0:
        return 1.0 - eps
    return value


def _ensure_gymnasium() -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            'Install dependencies: pip install "mujoco-gym-openenv"'
        ) from exc
    return gym


# Evaluation order + per-task grader scaling (return is domain-specific).
_TASK_ORDER = [
    "inverted_pendulum_v5",
    "hopper_v5",
    "halfcheetah_v5",
]

_TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "inverted_pendulum_v5": {
        "gym_id": "InvertedPendulum-v5",
        "difficulty": Difficulty.EASY,
        "description": "Balance an inverted pendulum on a sliding cart (classic continuous control).",
        "objective": "Keep the pole upright and maximize undiscounted return.",
        "rank": "easy",
        "return_offset": 15.0,
        "return_scale": 120.0,
    },
    "hopper_v5": {
        "gym_id": "Hopper-v5",
        "difficulty": Difficulty.MEDIUM,
        "description": "One-legged hopper: stay upright and move forward without falling.",
        "objective": "Apply torques so the hopper remains stable across the episode horizon.",
        "rank": "medium",
        "return_offset": 40.0,
        "return_scale": 900.0,
    },
    "halfcheetah_v5": {
        "gym_id": "HalfCheetah-v5",
        "difficulty": Difficulty.HARD,
        "description": "High-dimensional cheetah runner: coordinated multi-joint locomotion.",
        "objective": "Maximize forward velocity and return while staying within the physics simulator.",
        "rank": "hard",
        "return_offset": 0.0,
        "return_scale": 2800.0,
    },
}


class MuJoCoGymEnvironment(Environment[MuJoCoAction, MuJoCoObservation, MuJoCoState]):
    """Gymnasium MuJoCo tasks behind the OpenEnv Environment API."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, default_task_id: str = "inverted_pendulum_v5") -> None:
        super().__init__()
        if default_task_id not in _TASK_REGISTRY:
            raise KeyError(f"Unknown task_id: {default_task_id}")
        self._selected_task_id = default_task_id
        self._gym_env: Any | None = None
        self._state: MuJoCoState | None = None
        self._last_obs: list[float] | None = None
        self._last_info: dict[str, Any] = {}
        self.reset()

    def select_task(self, task_id: str) -> None:
        if task_id not in _TASK_REGISTRY:
            raise KeyError(f"Unknown task_id: {task_id}")
        self._selected_task_id = task_id

    def list_tasks(self) -> list[TaskDescriptor]:
        out: list[TaskDescriptor] = []
        for task_id in _TASK_ORDER:
            meta = _TASK_REGISTRY[task_id]
            out.append(
                TaskDescriptor(
                    metadata=TaskMetadata(
                        task_id=task_id,
                        difficulty=meta["difficulty"],
                        description=meta["description"],
                        objective=meta["objective"],
                    ),
                    expected_difficulty_rank=meta["rank"],
                )
            )
        return out

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> MuJoCoObservation:
        gym = _ensure_gymnasium()
        if task_id is not None:
            self.select_task(task_id)

        if self._gym_env is not None:
            self._gym_env.close()

        import numpy as np

        spec = _TASK_REGISTRY[self._selected_task_id]
        self._gym_env = gym.make(spec["gym_id"])
        obs, info = self._gym_env.reset(seed=seed)
        self._last_obs = np.asarray(obs, dtype=np.float64).ravel().tolist()
        self._last_info = dict(info or {})

        max_episode_steps = int(getattr(self._gym_env.spec, "max_episode_steps", 1000) or 1000)
        action_dim = int(np.prod(self._gym_env.action_space.shape))

        self._state = MuJoCoState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._selected_task_id,
            env_id=spec["gym_id"],
            done=False,
            cumulative_reward=0.0,
            terminated=False,
            truncated=False,
            action_dim=action_dim,
            obs_dim=len(self._last_obs),
            max_episode_steps=max_episode_steps,
            invalid_action_count=0,
            current_score=0.0,
        )
        grader = self.grade_current_episode()
        self._state.current_score = grader.score
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: MuJoCoAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> MuJoCoObservation:
        state = self._require_state()
        env = self._require_gym()

        if state.done:
            return self._build_observation(reward=0.0, done=True)

        state.step_count += 1
        expected = state.action_dim
        ctrl = action.control or []
        if len(ctrl) != expected:
            state.invalid_action_count += 1
            reward = -0.1
            state.cumulative_reward = round(state.cumulative_reward + reward, 6)
            grader = self.grade_current_episode()
            state.current_score = grader.score
            return self._build_observation(reward=reward, done=False)

        import numpy as np

        act_arr = np.asarray(ctrl, dtype=np.float32).reshape(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(act_arr)
        self._last_obs = np.asarray(obs, dtype=np.float64).ravel().tolist()
        self._last_info = dict(info or {})

        state.terminated = bool(terminated)
        state.truncated = bool(truncated)
        state.done = state.terminated or state.truncated
        state.cumulative_reward = round(state.cumulative_reward + float(reward), 6)

        grader = self.grade_current_episode()
        state.current_score = grader.score
        return self._build_observation(reward=float(reward), done=state.done)

    @property
    def state(self) -> MuJoCoState:
        s = self._require_state()
        s.current_score = self.grade_current_episode().score
        return s

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="mujoco_gym_env",
            description=(
                "Production-style MuJoCo continuous control via Gymnasium v5: multi-task suite "
                "(pendulum, hopper, half-cheetah), vector observations, boxed torques, dense rewards, "
                "and a deterministic multi-signal grader for agent evaluation."
            ),
            version="1.0.0",
            author="mujoco-gym-openenv",
        )

    def close(self) -> None:
        if self._gym_env is not None:
            self._gym_env.close()
            self._gym_env = None

    def grade_current_episode(self) -> GraderResult:
        state = self._require_state()
        task_id = state.task_id
        spec = _TASK_REGISTRY[task_id]
        ret = state.cumulative_reward
        steps = state.step_count
        horizon = max(1, state.max_episode_steps)

        length_score = min(1.0, steps / max(1.0, horizon * 0.4))
        ro = float(spec["return_offset"])
        rs = float(spec["return_scale"])
        return_score = min(1.0, max(0.0, (ret + ro) / rs))
        combined = 0.35 * length_score + 0.65 * return_score
        if state.terminated and not state.truncated:
            combined *= 0.45
        inv_penalty = min(0.25, 0.06 * state.invalid_action_count)
        raw = max(0.0, min(1.0, combined - inv_penalty))
        score = round(_strict_unit_interval(raw), 6)

        breakdown = {
            "length_progress": round(_strict_unit_interval(length_score), 6),
            "return_progress": round(_strict_unit_interval(return_score), 6),
            "episode_done": round(_strict_unit_interval(1.0 if state.done else 0.25), 6),
        }
        passed = score >= 0.8
        return GraderResult(task_id=task_id, score=score, breakdown=breakdown, passed=passed)

    def _build_observation(self, reward: float, done: bool) -> MuJoCoObservation:
        state = self._require_state()
        obs_list = list(self._last_obs) if self._last_obs is not None else []
        return MuJoCoObservation(
            task_id=state.task_id,
            env_id=state.env_id,
            obs=obs_list,
            obs_dim=state.obs_dim,
            action_dim=state.action_dim,
            max_episode_steps=state.max_episode_steps,
            step_count=state.step_count,
            last_info=self._last_info,
            done=done,
            reward=reward,
        )

    def _require_state(self) -> MuJoCoState:
        if self._state is None:
            raise RuntimeError("State not initialized.")
        return self._state

    def _require_gym(self) -> Any:
        if self._gym_env is None:
            raise RuntimeError("Gymnasium env not initialized; call reset() first.")
        return self._gym_env
