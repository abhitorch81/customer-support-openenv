from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from ._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    LOOKUP_ORDER = "lookup_order"
    LOOKUP_POLICY = "lookup_policy"
    ASK_CUSTOMER = "ask_customer"
    SET_ISSUE_TYPE = "set_issue_type"
    SET_PRIORITY = "set_priority"
    DECIDE_RESOLUTION = "decide_resolution"
    ESCALATE_TICKET = "escalate_ticket"
    CLOSE_TICKET = "close_ticket"


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action_type: ActionType
    argument: str | None = None
    notes: str | None = None
    outcome: str


class SupportAction(Action):
    action_type: ActionType
    argument: str | None = None
    notes: str | None = None


class SupportObservation(Observation):
    task_id: str
    difficulty: Difficulty
    ticket_id: str
    objective: str
    customer_message: str
    available_actions: list[ActionType]
    visible_order: dict[str, Any]
    visible_policy: str | None = None
    revealed_customer_details: dict[str, str] = Field(default_factory=dict)
    known_missing_fields: list[str] = Field(default_factory=list)
    history: list[ActionRecord] = Field(default_factory=list)
    last_action_result: str = ""
    step_count: int = 0
    max_steps: int = 12
    cumulative_reward: float = 0.0


class SupportState(State):
    task_id: str
    difficulty: Difficulty
    ticket_id: str
    max_steps: int
    done: bool
    cumulative_reward: float
    invalid_action_count: int
    order_retrieved: bool
    policy_retrieved: bool
    required_info_asked: list[str]
    revealed_customer_details: dict[str, str]
    issue_type_guess: str | None = None
    priority_guess: str | None = None
    resolution_guess: str | None = None
    escalated: bool = False
    closed: bool = False
    premature_resolution_attempted: bool = False
    wrong_resolution_attempted: bool = False
    bad_close_attempted: bool = False
    unnecessary_escalation_attempted: bool = False
    history: list[ActionRecord] = Field(default_factory=list)
    current_score: float = Field(ge=0.0, le=1.0)


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float]
    passed: bool


class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    description: str
    objective: str


class TaskDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: TaskMetadata
    expected_difficulty_rank: Literal["easy", "medium", "hard"]


class TasksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskDescriptor]
    action_schema: dict[str, Any]


class BaselineEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    steps: int
    total_reward: float
    policy_name: str


class BaselineRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_name: str
    model_name: str | None = None
    average_score: float = Field(ge=0.0, le=1.0)
    results: list[BaselineEpisodeResult]
