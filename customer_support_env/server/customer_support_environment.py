from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from .._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..catalog import load_task_catalog
from ..models import (
    ActionRecord,
    ActionType,
    Difficulty,
    GraderResult,
    SupportAction,
    SupportObservation,
    SupportState,
    TaskDescriptor,
    TaskMetadata,
)


def _strict_unit_interval(value: float) -> float:
    """Return a score strictly inside (0, 1) for validator compatibility."""
    eps = 1e-6
    if value <= 0.0:
        return eps
    if value >= 1.0:
        return 1.0 - eps
    return value


class SupportTicketEnvironment(
    Environment[SupportAction, SupportObservation, SupportState]
):
    """Official OpenEnv environment for customer support ticket resolution."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, default_task_id: str = "easy_damaged_mug") -> None:
        super().__init__()
        self._catalog = {task["task_id"]: task for task in load_task_catalog()}
        self._selected_task_id = default_task_id
        self._task: dict[str, Any] | None = None
        self._state: SupportState | None = None
        self._reset_message = "Environment ready."
        self.reset()

    def select_task(self, task_id: str) -> None:
        if task_id not in self._catalog:
            raise KeyError(f"Unknown task_id: {task_id}")
        self._selected_task_id = task_id

    def list_tasks(self) -> list[TaskDescriptor]:
        ordered = ["easy_damaged_mug", "medium_wrong_hoodie", "hard_laptop_policy_exception"]
        descriptors: list[TaskDescriptor] = []
        for task_id in ordered:
            task = self._catalog[task_id]
            descriptors.append(
                TaskDescriptor(
                    metadata=TaskMetadata(
                        task_id=task["task_id"],
                        difficulty=Difficulty(task["difficulty"]),
                        description=task["description"],
                        objective=task["objective"],
                    ),
                    expected_difficulty_rank=task["difficulty"],
                )
            )
        return descriptors

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> SupportObservation:
        if task_id is not None:
            self.select_task(task_id)

        task = copy.deepcopy(self._catalog[self._selected_task_id])
        self._task = task
        self._state = SupportState(
            episode_id=episode_id or str(uuid4()),
            task_id=task["task_id"],
            difficulty=Difficulty(task["difficulty"]),
            ticket_id=task["ticket_id"],
            step_count=0,
            max_steps=12,
            done=False,
            cumulative_reward=0.0,
            invalid_action_count=0,
            order_retrieved=False,
            policy_retrieved=False,
            required_info_asked=[],
            revealed_customer_details={},
            issue_type_guess=None,
            priority_guess=None,
            resolution_guess=None,
            escalated=False,
            closed=False,
            premature_resolution_attempted=False,
            wrong_resolution_attempted=False,
            bad_close_attempted=False,
            unnecessary_escalation_attempted=False,
            history=[],
            current_score=0.0,
        )
        return self._build_observation(last_action_result=self._reset_message, reward=0.0)

    def step(
        self,
        action: SupportAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> SupportObservation:
        state = self._require_state()
        task = self._require_task()

        if state.done:
            return self._build_observation(
                last_action_result="Episode already finished. Reset before taking more actions.",
                reward=0.0,
            )

        state.step_count += 1
        reward_components: dict[str, float] = {"time_penalty": -0.01}
        reward_value = -0.01

        match action.action_type:
            case ActionType.LOOKUP_ORDER:
                outcome = self._handle_lookup_order(task, state, reward_components)
            case ActionType.LOOKUP_POLICY:
                outcome = self._handle_lookup_policy(state, reward_components)
            case ActionType.ASK_CUSTOMER:
                outcome = self._handle_ask_customer(task, state, action, reward_components)
            case ActionType.SET_ISSUE_TYPE:
                outcome = self._handle_set_issue_type(task, state, action, reward_components)
            case ActionType.SET_PRIORITY:
                outcome = self._handle_set_priority(task, state, action, reward_components)
            case ActionType.DECIDE_RESOLUTION:
                outcome = self._handle_decide_resolution(task, state, action, reward_components)
            case ActionType.ESCALATE_TICKET:
                outcome = self._handle_escalate(task, state, reward_components)
            case ActionType.CLOSE_TICKET:
                outcome = self._handle_close(task, state, reward_components)

        reward_value += sum(value for key, value in reward_components.items() if key != "time_penalty")

        if state.step_count > 8:
            overtime_penalty = -0.02
            reward_components["overtime_penalty"] = overtime_penalty
            reward_value += overtime_penalty

        if state.step_count >= state.max_steps and not state.done:
            state.done = True
            reward_components["max_steps_penalty"] = -0.10
            reward_value += -0.10
            outcome = f"{outcome} Maximum step budget reached."

        state.cumulative_reward = round(state.cumulative_reward + reward_value, 4)
        state.history.append(
            ActionRecord(
                step_index=state.step_count,
                action_type=action.action_type,
                argument=action.argument,
                notes=action.notes,
                outcome=outcome,
            )
        )

        grader = self.grade_current_episode()
        state.current_score = grader.score
        return self._build_observation(last_action_result=outcome, reward=reward_value)

    @property
    def state(self) -> SupportState:
        state = self._require_state()
        state.current_score = self.grade_current_episode().score
        return state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="customer_support_env",
            description="Customer support ticket resolution environment with deterministic graders.",
            version="0.2.0",
            author="Codex + user",
        )

    def close(self) -> None:
        return None

    def grade_current_episode(self) -> GraderResult:
        state = self._require_state()
        task = self._require_task()
        truth = task["ground_truth"]

        required = truth["required_info"]
        asked_count = sum(1 for item in required if item in state.required_info_asked)
        required_info_score = 1.0 if not required else asked_count / len(required)

        breakdown = {
            "order_lookup": 1.0 if state.order_retrieved else 0.0,
            "policy_lookup": 1.0 if state.policy_retrieved else 0.0,
            "required_info": round(required_info_score, 4),
            "issue_type": 1.0 if state.issue_type_guess == truth["issue_type"] else 0.0,
            "priority": 1.0 if state.priority_guess == truth["priority"] else 0.0,
            "resolution": 1.0 if state.resolution_guess == truth["resolution"] else 0.0,
            "escalation": 1.0 if state.escalated == truth["escalation_required"] else 0.0,
            "closed": 1.0 if state.closed else 0.0,
            "workflow_discipline": round(self._workflow_discipline_score(state), 4),
        }

        score = (
            0.08 * breakdown["order_lookup"]
            + 0.10 * breakdown["policy_lookup"]
            + 0.18 * breakdown["required_info"]
            + 0.12 * breakdown["issue_type"]
            + 0.10 * breakdown["priority"]
            + 0.22 * breakdown["resolution"]
            + 0.08 * breakdown["escalation"]
            + 0.04 * breakdown["closed"]
            + 0.08 * breakdown["workflow_discipline"]
        )

        step_budget = self._score_step_budget(state.difficulty)
        penalty = min(
            0.30,
            0.04 * state.invalid_action_count + 0.015 * max(0, state.step_count - step_budget),
        )
        score = max(0.0, min(1.0, round(score - penalty, 4)))
        score = round(_strict_unit_interval(score), 6)

        return GraderResult(
            task_id=state.task_id,
            score=score,
            breakdown=breakdown,
            passed=score >= 0.8,
        )

    def _build_observation(self, last_action_result: str, reward: float) -> SupportObservation:
        state = self._require_state()
        task = self._require_task()

        visible_order = copy.deepcopy(task["initial_order_view"])
        if state.order_retrieved:
            visible_order = copy.deepcopy(task["order"])

        return SupportObservation(
            task_id=task["task_id"],
            difficulty=Difficulty(task["difficulty"]),
            ticket_id=task["ticket_id"],
            objective=task["objective"],
            customer_message=task["customer_message"],
            available_actions=list(ActionType),
            visible_order=visible_order,
            visible_policy=task["policy"] if state.policy_retrieved else None,
            revealed_customer_details=copy.deepcopy(state.revealed_customer_details),
            known_missing_fields=self._remaining_required_fields(task, state),
            history=copy.deepcopy(state.history),
            last_action_result=last_action_result,
            step_count=state.step_count,
            max_steps=state.max_steps,
            cumulative_reward=round(state.cumulative_reward, 4),
            done=state.done,
            reward=round(reward, 4),
        )

    def _handle_lookup_order(
        self,
        task: dict[str, Any],
        state: SupportState,
        reward_components: dict[str, float],
    ) -> str:
        if state.order_retrieved:
            reward_components["repeat_lookup_order"] = -0.04
            return "Order record was already opened."
        state.order_retrieved = True
        reward_components["order_lookup"] = 0.12
        return (
            f"Opened order record for {task['order']['order_id']}. "
            f"Days since delivery: {task['order']['days_since_delivery']}."
        )

    def _handle_lookup_policy(self, state: SupportState, reward_components: dict[str, float]) -> str:
        if state.policy_retrieved:
            reward_components["repeat_lookup_policy"] = -0.04
            return "Policy was already opened."
        state.policy_retrieved = True
        reward_components["policy_lookup"] = 0.10
        return "Opened the relevant return and escalation policy."

    def _handle_ask_customer(
        self,
        task: dict[str, Any],
        state: SupportState,
        action: SupportAction,
        reward_components: dict[str, float],
    ) -> str:
        if not action.argument:
            reward_components["invalid_question"] = -0.08
            state.invalid_action_count += 1
            return "ask_customer requires an argument naming the requested field."

        customer_data = task["customer_data"]
        if action.argument not in customer_data:
            reward_components["irrelevant_question"] = -0.05
            return f"The customer has no new information for '{action.argument}'."

        if action.argument in state.revealed_customer_details:
            reward_components["repeat_question"] = -0.03
            return f"The field '{action.argument}' was already collected."

        state.revealed_customer_details[action.argument] = customer_data[action.argument]
        if action.argument in task["ground_truth"]["required_info"]:
            state.required_info_asked.append(action.argument)
            reward_components["required_info"] = 0.15
        else:
            reward_components["optional_info"] = 0.02
        return f"Collected customer verification field '{action.argument}'."

    def _handle_set_issue_type(
        self,
        task: dict[str, Any],
        state: SupportState,
        action: SupportAction,
        reward_components: dict[str, float],
    ) -> str:
        if not action.argument:
            reward_components["invalid_issue_type"] = -0.08
            state.invalid_action_count += 1
            return "set_issue_type requires a string argument."

        state.issue_type_guess = action.argument
        if action.argument == task["ground_truth"]["issue_type"]:
            reward_components["correct_issue_type"] = 0.15
            return f"Issue type set correctly to '{action.argument}'."
        reward_components["wrong_issue_type"] = -0.06
        return f"Issue type '{action.argument}' does not match the ticket evidence."

    def _handle_set_priority(
        self,
        task: dict[str, Any],
        state: SupportState,
        action: SupportAction,
        reward_components: dict[str, float],
    ) -> str:
        if not action.argument:
            reward_components["invalid_priority"] = -0.08
            state.invalid_action_count += 1
            return "set_priority requires a string argument."

        state.priority_guess = action.argument
        if action.argument == task["ground_truth"]["priority"]:
            reward_components["correct_priority"] = 0.10
            return f"Priority set correctly to '{action.argument}'."
        reward_components["wrong_priority"] = -0.05
        return f"Priority '{action.argument}' is not appropriate for this ticket."

    def _handle_decide_resolution(
        self,
        task: dict[str, Any],
        state: SupportState,
        action: SupportAction,
        reward_components: dict[str, float],
    ) -> str:
        if not action.argument:
            reward_components["invalid_resolution"] = -0.08
            state.invalid_action_count += 1
            return "decide_resolution requires a string argument."

        state.resolution_guess = action.argument
        if not self._required_info_satisfied(task, state):
            state.premature_resolution_attempted = True
            reward_components["premature_resolution"] = -0.20
            return "Resolution chosen before all required customer verification was collected."

        if action.argument == task["ground_truth"]["resolution"]:
            reward_components["correct_resolution"] = 0.35
            return f"Resolution set correctly to '{action.argument}'."

        state.wrong_resolution_attempted = True
        reward_components["wrong_resolution"] = -0.22
        return f"Resolution '{action.argument}' is incorrect for this case."

    def _handle_escalate(
        self,
        task: dict[str, Any],
        state: SupportState,
        reward_components: dict[str, float],
    ) -> str:
        if state.escalated:
            reward_components["repeat_escalation"] = -0.03
            return "Ticket was already escalated."

        state.escalated = True
        if task["ground_truth"]["escalation_required"]:
            reward_components["correct_escalation"] = 0.14
            return "Ticket escalated to the manual review queue."

        state.unnecessary_escalation_attempted = True
        reward_components["unnecessary_escalation"] = -0.08
        return "Escalation was unnecessary for this ticket."

    def _handle_close(
        self,
        task: dict[str, Any],
        state: SupportState,
        reward_components: dict[str, float],
    ) -> str:
        if state.closed:
            reward_components["repeat_close"] = -0.03
            return "Ticket is already closed."

        state.closed = True
        state.done = True
        if self._can_close_successfully(task, state):
            reward_components["successful_close"] = 0.12
            return "Ticket closed successfully with a valid resolution path."

        state.bad_close_attempted = True
        reward_components["bad_close"] = -0.18
        return "Ticket was closed before the workflow was correctly completed."

    def _required_info_satisfied(self, task: dict[str, Any], state: SupportState) -> bool:
        required = task["ground_truth"]["required_info"]
        return all(field in state.required_info_asked for field in required)

    def _remaining_required_fields(self, task: dict[str, Any], state: SupportState) -> list[str]:
        required = task["ground_truth"]["required_info"]
        return [field for field in required if field not in state.required_info_asked]

    def _can_close_successfully(self, task: dict[str, Any], state: SupportState) -> bool:
        truth = task["ground_truth"]
        return all(
            [
                self._required_info_satisfied(task, state),
                state.issue_type_guess == truth["issue_type"],
                state.priority_guess == truth["priority"],
                state.resolution_guess == truth["resolution"],
                state.escalated == truth["escalation_required"],
            ]
        )

    def _workflow_discipline_score(self, state: SupportState) -> float:
        issues = sum(
            [
                state.premature_resolution_attempted,
                state.wrong_resolution_attempted,
                state.bad_close_attempted,
                state.unnecessary_escalation_attempted,
            ]
        )
        return max(0.0, 1.0 - 0.34 * issues)

    def _score_step_budget(self, difficulty: Difficulty) -> int:
        if difficulty == Difficulty.EASY:
            return 6
        if difficulty == Difficulty.MEDIUM:
            return 8
        return 9

    def _require_state(self) -> SupportState:
        if self._state is None:
            raise RuntimeError("State not initialized.")
        return self._state

    def _require_task(self) -> dict[str, Any]:
        if self._task is None:
            raise RuntimeError("Task not initialized.")
        return self._task
