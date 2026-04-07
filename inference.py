from __future__ import annotations

import os
import sys

from customer_support_env.baseline import run_baseline


def main() -> None:
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
    try:
        result = run_baseline(policy_name="openai", model_name=model_name)
    except Exception as exc:
        # Phase fail-fast safety: ensure we always output a valid JSON payload.
        # If OpenAI fails for any reason (auth/network/response), fall back to heuristic.
        print(f"[inference] OpenAI baseline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        result = run_baseline(policy_name="heuristic", model_name=None)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
