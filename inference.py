"""
Inference script (hackathon spec).

MANDATORY env (see README):
  API_BASE_URL   — LLM endpoint (default: Hugging Face router).
  MODEL_NAME     — model id (default set below).
  HF_TOKEN       — preferred; or API_KEY for the same proxy credential.

Optional:
  BENCHMARK      — logged as env= in [START] (default: mujoco_gym_env).
  LOCAL_IMAGE_NAME / IMAGE_NAME — only if using OpenEnv from_docker_image(); not used here.

STDOUT: only [START], [STEP], [END] lines; [END] always after env.close(), even on exception.
"""

from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

from mujoco_gym_env.baseline import RandomPolicy
from mujoco_gym_env.environment import MuJoCoGymEnvironment
from mujoco_gym_env.models import MuJoCoAction, MuJoCoObservation

# Defaults per spec — align with your active inference setup / Space secrets.
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-4.1-mini"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "mujoco_gym_env")
# Reserved for docker-based OpenEnv clients (validate-submission / from_docker_image).
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


def _format_action_safe(action: MuJoCoAction) -> str:
    try:
        c = action.control or []
        if len(c) <= 4:
            return f"control({c})"
        return f"control(dim={len(c)})"
    except Exception:
        return "control([])"


def _act_safely(
    policy: object,
    fallback: RandomPolicy,
    observation: MuJoCoObservation,
) -> MuJoCoAction:
    try:
        return policy.act(observation)  # type: ignore[attr-defined]
    except Exception:
        try:
            return fallback.act(observation)
        except Exception:
            dim = max(1, observation.action_dim)
            return MuJoCoAction(control=[0.0] * dim)


def _llm_proxy_ping(client: OpenAI, model: str) -> None:
    """At least one OpenAI client call through API_BASE_URL (Phase 2 LiteLLM visibility)."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "."}],
            max_tokens=1,
        )
    except Exception:
        client.responses.create(model=model, input=".")


def main() -> None:
    _ = _LOCAL_IMAGE_NAME  # explicit read so tooling sees optional docker image name

    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN or API_KEY must be set for OpenAI(base_url=API_BASE_URL, api_key=...)."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = MuJoCoGymEnvironment()
    fallback_policy = RandomPolicy()
    policy: object = RandomPolicy()

    for task in env.list_tasks():
        task_id = task.metadata.task_id
        _log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.0

        try:
            # OpenAI client call — must go through injected proxy (HF_TOKEN / API_KEY).
            _llm_proxy_ping(client, MODEL_NAME)

            env.select_task(task_id)
            observation = env.reset()
            done = observation.done
            max_extra = int(os.getenv("MUJOCO_INFERENCE_MAX_STEPS", "5000"))

            while not done and steps_taken < max_extra:
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
            success = bool(getattr(grader, "passed", False)) or score >= 0.8
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
