from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from ._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    ADD_GROUP = "add_group"
    REPLACE_SUBSTRUCTURE = "replace_substructure"
    REMOVE_GROUP = "remove_group"
    BIOISOSTERE_SWAP = "bioisostere_swap"
    SCORE_MOLECULE = "score_molecule"
    COMPARE_CANDIDATES = "compare_candidates"
    STOP_AND_SUBMIT = "stop_and_submit"


class DrugDiscoveryAction(Action):
    action_type: ActionType
    query_smarts: str | None = None
    replacement_smiles: str | None = None
    group_key: str | None = Field(
        default=None,
        description="For add_group: canned key e.g. methyl_aromatic, fluoro_aromatic",
    )
    bioisostere_key: str | None = Field(
        default=None,
        description="For bioisostere_swap: e.g. hydroxyl_to_fluoro, nitro_to_amino",
    )
    candidate_indices: list[int] = Field(default_factory=list)
    notes: str | None = None


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


class DrugDiscoveryObservation(Observation):
    task_id: str
    difficulty: Difficulty
    smiles: str
    descriptors: dict[str, float]
    descriptor_text: str
    target_profile_summary: str
    modification_history: list[str]
    candidate_pool: list[str]
    step_count: int
    max_steps: int
    best_score_so_far: float
    best_smiles: str
    available_actions: list[ActionType]
    last_action_result: str


class DrugDiscoveryState(State):
    task_id: str
    difficulty: Difficulty
    initial_murcko_smiles: str
    smiles: str
    done: bool
    cumulative_reward: float
    step_count: int
    max_steps: int
    modification_history: list[str]
    candidate_pool: list[str]
    best_smiles: str
    best_score: float = Field(gt=0.0, lt=1.0)
    last_descriptors: dict[str, float]
    scored_this_episode: bool
    submitted: bool
    invalid_action_count: int
    current_score: float = Field(gt=0.0, lt=1.0)


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(gt=0.0, lt=1.0)
    breakdown: dict[str, float]
    passed: bool
    rationale: str = ""

    @field_validator("breakdown")
    @classmethod
    def _breakdown_open_unit_interval(cls, d: dict[str, float]) -> dict[str, float]:
        for key, v in d.items():
            x = float(v)
            if not (0.0 < x < 1.0):
                raise ValueError(f"breakdown[{key!r}] must be strictly in (0, 1), got {x!r}")
        return d
