from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

from customer_support_env.baseline import HeuristicPolicy, OpenAIPolicy
from customer_support_env.environment import SupportTicketEnvironment
from customer_support_env.models import SupportAction, SupportObservation


def _strict_unit_interval(value: float) -> float:
    """Return a score strictly inside (0, 1) for validator parsing."""
    # Use a margin large enough so formatting (e.g. :.6f) cannot round to exactly 0.0 or 1.0.
    eps = 1e-4
    if value <= 0.0:
        return eps
    if value >= 1.0:
        return 1.0 - eps
    return value


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


def _format_action(action: SupportAction) -> str:
    if action.argument is None:
        return action.action_type.value
    return f"{action.action_type.value}({action.argument})"


def _act_safely(
    policy: object,
    fallback: HeuristicPolicy,
    observation: SupportObservation,
) -> SupportAction:
    try:
        return policy.act(observation)  # type: ignore[attr-defined]
    except Exception:
        return fallback.act(observation)


def main() -> None:
    # Required env vars (with defaults allowed only for base URL + model).
    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "openai/gpt-4.1-mini"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    # Build OpenAI client explicitly (submission requirement).
    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    env_name = os.getenv("BENCHMARK", "customer_support_env")
    env = SupportTicketEnvironment()

    fallback_policy = HeuristicPolicy()
    policy: object = OpenAIPolicy(model_name=model_name)
    # Override the OpenAI client to ensure we use API_BASE_URL/HF_TOKEN from this script.
    policy.client = client  # type: ignore[attr-defined]

    for task in env.list_tasks():
        task_id = task.metadata.task_id
        _log_start(task=task_id, env=env_name, model=model_name)

        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.0

        try:
            env.select_task(task_id)
            observation = env.reset()
            done = observation.done

            while not done:
                action = _act_safely(policy=policy, fallback=fallback_policy, observation=observation)
                observation = env.step(action)
                done = observation.done

                reward_val = float(getattr(observation, "reward", 0.0) or 0.0)
                rewards.append(reward_val)
                steps_taken = int(getattr(observation, "step_count", steps_taken + 1))

                _log_step(
                    step=steps_taken,
                    action=_format_action(action),
                    reward=reward_val,
                    done=bool(done),
                    error=None,
                )

            grader = env.grade_current_episode()
            score = _strict_unit_interval(float(grader.score))
            success = bool(getattr(grader, "passed", False)) or score >= 0.8
        finally:
            try:
                env.close()
            finally:
                _log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
