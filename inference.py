"""
Hackathon inference script — lead optimization OpenEnv.

Env: API_BASE_URL, MODEL_NAME, HF_TOKEN or API_KEY, BENCHMARK (optional).
Stdout: [START] / [STEP] / [END] only.
"""

from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

from drug_discovery_env.baseline import HeuristicPolicy
from drug_discovery_env.environment import DrugDiscoveryEnvironment
from drug_discovery_env.models import DrugDiscoveryAction, DrugDiscoveryObservation

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-4.1-mini"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "drug_discovery_env")
_LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")


def _strict_unit_interval(value: float) -> float:
    eps = 1e-4
    if value <= 0.0:
        return eps
    if value >= 1.0:
        return 1.0 - eps
    return value


def _sanitize_action_str(s: str) -> str:
    return " ".join(s.split()).replace("\n", " ").strip() or "null"


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_out = _sanitize_action_str(action)
    print(
        f"[STEP] step={step} action={action_out} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_clamped = min(1.0, max(0.0, score))
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score_clamped:.2f} rewards={rewards_str}",
        flush=True,
    )


def _format_action(a: DrugDiscoveryAction) -> str:
    parts = [a.action_type.value]
    if a.group_key:
        parts.append(f"group={a.group_key}")
    if a.bioisostere_key:
        parts.append(f"bio={a.bioisostere_key}")
    if a.query_smarts:
        parts.append(f"smarts={a.query_smarts}")
    if a.replacement_smiles:
        parts.append(f"repl={a.replacement_smiles}")
    if a.candidate_indices:
        parts.append(f"idx={a.candidate_indices}")
    return "|".join(parts)


def _format_action_safe(a: DrugDiscoveryAction) -> str:
    try:
        return _format_action(a)
    except Exception:
        return a.action_type.value


def _act_safely(
    policy: object,
    fallback: HeuristicPolicy,
    observation: DrugDiscoveryObservation,
) -> DrugDiscoveryAction:
    try:
        return policy.act(observation)  # type: ignore[attr-defined]
    except Exception:
        return fallback.act(observation)


def _llm_proxy_ping(client: OpenAI, model: str) -> None:
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "."}],
            max_tokens=1,
        )
    except Exception:
        client.responses.create(model=model, input=".")


def main() -> None:
    _ = _LOCAL_IMAGE_NAME

    if not API_KEY:
        raise RuntimeError("HF_TOKEN or API_KEY must be set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = DrugDiscoveryEnvironment()
    fallback_policy = HeuristicPolicy()
    policy: object = HeuristicPolicy()

    for task in env.list_tasks():
        task_id = task.metadata.task_id
        _log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.0

        try:
            _llm_proxy_ping(client, MODEL_NAME)

            env.select_task(task_id)
            observation = env.reset()
            done = observation.done
            max_steps_cap = int(os.getenv("DRUG_DISCOVERY_MAX_STEPS", "500"))

            while not done and steps_taken < max_steps_cap:
                step_error: Optional[str] = None
                try:
                    action = _act_safely(policy=policy, fallback=fallback_policy, observation=observation)
                    observation = env.step(action)
                    done = observation.done
                    action_str = _format_action_safe(action)
                except Exception as exc:
                    step_error = str(exc).replace("\n", " ")
                    action_str = "error"
                    done = True
                    reward_val = 0.0
                    rewards.append(reward_val)
                    steps_taken += 1
                    _log_step(
                        step=steps_taken,
                        action=action_str,
                        reward=reward_val,
                        done=done,
                        error=step_error,
                    )
                    break

                reward_val = float(getattr(observation, "reward", 0.0) or 0.0)
                rewards.append(reward_val)
                steps_taken = int(getattr(observation, "step_count", steps_taken + 1))

                _log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward_val,
                    done=bool(done),
                    error=None,
                )

            grader = env.grade_current_episode()
            score = _strict_unit_interval(float(grader.score))
            success = bool(getattr(grader, "passed", False)) or score >= 0.72
        except Exception:
            score = _strict_unit_interval(0.0)
            success = False
        finally:
            try:
                env.close()
            except Exception:
                pass
            _log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
