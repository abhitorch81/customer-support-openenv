from __future__ import annotations

import os

from customer_support_env.baseline import run_baseline


def main() -> None:
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
    result = run_baseline(policy_name="openai", model_name=model_name)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
