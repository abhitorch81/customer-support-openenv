from __future__ import annotations

from typing import Any, Literal

from ._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

from customer_support_env.models import Difficulty, TaskMetadata


class MuJoCoAction(Action):
    """Continuous control; length must match the active task's action dimension at step time."""

    control: list[float] = Field(default_factory=list)


class MuJoCoObservation(Observation):
    task_id: str
    env_id: str
    obs: list[float]
    obs_dim: int
    action_dim: int
    max_episode_steps: int
    step_count: int = 0
    last_info: dict[str, Any] = Field(default_factory=dict)


class MuJoCoState(State):
    task_id: str
    env_id: str
    done: bool
    cumulative_reward: float
    terminated: bool
    truncated: bool
    action_dim: int
    obs_dim: int
    max_episode_steps: int
    invalid_action_count: int
    current_score: float = Field(ge=0.0, le=1.0)


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float]
    passed: bool


class TaskDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: TaskMetadata
    expected_difficulty_rank: Literal["easy", "medium", "hard"]


class TasksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskDescriptor]
    action_schema: dict[str, Any]
