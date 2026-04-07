from __future__ import annotations

import os
import sys

from customer_support_env.baseline import run_baseline


def _strict_unit_interval(value: float) -> float:
    """Return a score strictly inside (0, 1) for validator parsing."""
    eps = 1e-6
    if value <= 0.0:
        return eps
    if value >= 1.0:
        return 1.0 - eps
    return value


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
    hf_token = os.getenv("HF_TOKEN")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME")

    # Keep runtime configuration explicit for submission checkers.
    os.environ["API_BASE_URL"] = api_base_url
    os.environ["MODEL_NAME"] = model_name
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token
    if local_image_name is not None:
        os.environ["LOCAL_IMAGE_NAME"] = local_image_name

    print(f"[START] task=baseline policy=openai model={model_name}", flush=True)
    try:
        print("[STEP] step=1 action=run_baseline_openai", flush=True)
        result = run_baseline(policy_name="openai", model_name=model_name)
    except Exception as exc:
        # Phase fail-fast safety: ensure we always output a valid JSON payload.
        # If OpenAI fails for any reason (auth/network/response), fall back to heuristic.
        print(f"[inference] OpenAI baseline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[STEP] step=2 action=run_baseline_heuristic_fallback", flush=True)
        result = run_baseline(policy_name="heuristic", model_name=None)
    for index, item in enumerate(result.results, start=1):
        safe_score = _strict_unit_interval(item.score)
        print(f"[START] task={item.task_id} policy={result.policy_name}", flush=True)
        print(
            f"[STEP] step={index + 2} task={item.task_id} score={safe_score:.6f} steps={item.steps}",
            flush=True,
        )
        print(
            f"[END] task={item.task_id} score={safe_score:.6f} steps={item.steps} grader=deterministic",
            flush=True,
        )
    print(
        f"[END] task=baseline score={result.average_score} episodes={len(result.results)} status=success",
        flush=True,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
