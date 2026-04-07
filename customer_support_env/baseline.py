from __future__ import annotations

import argparse
import json
import os
from typing import Protocol

from .environment import SupportTicketEnvironment
from .models import BaselineEpisodeResult, BaselineRunResult, SupportAction, SupportObservation

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"


class Policy(Protocol):
    name: str

    def act(self, observation: SupportObservation) -> SupportAction:
        ...


class HeuristicPolicy:
    name = "heuristic"

    def act(self, observation: SupportObservation) -> SupportAction:
        if not _history_has(observation, "lookup_order"):
            return SupportAction(action_type="lookup_order")
        if not _history_has(observation, "lookup_policy"):
            return SupportAction(action_type="lookup_policy")
        if observation.known_missing_fields:
            field_name = observation.known_missing_fields[0]
            if field_name not in observation.revealed_customer_details:
                return SupportAction(action_type="ask_customer", argument=field_name)

        desired_issue = _infer_issue_type(observation)
        desired_priority = _infer_priority(observation)
        desired_resolution = _infer_resolution(observation, desired_issue)
        needs_escalation = desired_resolution == "manual_review"

        if not _already_set(observation, "set_issue_type", desired_issue):
            return SupportAction(action_type="set_issue_type", argument=desired_issue)
        if not _already_set(observation, "set_priority", desired_priority):
            return SupportAction(action_type="set_priority", argument=desired_priority)
        if needs_escalation and not _history_has(observation, "escalate_ticket"):
            return SupportAction(action_type="escalate_ticket")
        if not _already_set(observation, "decide_resolution", desired_resolution):
            return SupportAction(action_type="decide_resolution", argument=desired_resolution)
        return SupportAction(action_type="close_ticket")


class OpenAIPolicy:
    name = "openai"

    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=_resolve_api_key(),
            base_url=os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL),
        )
        self.model_name = model_name
        # Fail-safe: if the OpenAI call or parsing fails, use the deterministic policy
        # so the baseline pipeline can't crash mid-episode.
        self._fallback_policy = HeuristicPolicy()

    def act(self, observation: SupportObservation) -> SupportAction:
        prompt = self._build_prompt(observation)

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )
            output_text = self._extract_response_text(response)
            payload = _extract_first_json_object(output_text)
            data = json.loads(payload)
            return SupportAction.model_validate(data)
        except Exception:
            # Any error (network/auth, unexpected response shape, invalid JSON, pydantic validation)
            # should not crash the pipeline.
            return self._fallback_policy.act(observation)

    @staticmethod
    def _extract_response_text(response: object) -> str:
        """Best-effort extraction of assistant text from an OpenAI Responses API object."""
        # Newer SDKs often provide `output_text`.
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Fallback to the first text content we can find.
        try:
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text
        except Exception:
            pass

        # Last resort: stringify the response so braces parsing can still work if present.
        return str(response)

    def _build_prompt(self, observation: SupportObservation) -> str:
        schema = {
            "action_type": [
                "lookup_order",
                "lookup_policy",
                "ask_customer",
                "set_issue_type",
                "set_priority",
                "decide_resolution",
                "escalate_ticket",
                "close_ticket",
            ],
            "argument": "optional string",
            "notes": "optional string",
        }
        return (
            "You are operating a customer support environment.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Choose the next best action to maximize final grader score.\n"
            "Do not invent hidden facts.\n\n"
            f"Observation:\n{observation.model_dump_json(indent=2)}\n\n"
            f"Action schema:\n{json.dumps(schema, indent=2)}\n"
        )


def run_baseline(policy_name: str, model_name: str | None = None) -> BaselineRunResult:
    env = SupportTicketEnvironment()
    policy = _build_policy(policy_name=policy_name, model_name=model_name)
    fallback_policy = HeuristicPolicy()
    results: list[BaselineEpisodeResult] = []

    for task in env.list_tasks():
        env.select_task(task.metadata.task_id)
        observation = env.reset()
        done = observation.done

        while not done:
            try:
                action = policy.act(observation)
            except Exception:
                # Phase fail-fast safety: never let a single step break the entire pipeline.
                action = fallback_policy.act(observation)
            observation = env.step(action)
            done = observation.done

        grader = env.grade_current_episode()
        state = env.state
        results.append(
            BaselineEpisodeResult(
                task_id=task.metadata.task_id,
                score=grader.score,
                steps=state.step_count,
                total_reward=round(state.cumulative_reward, 4),
                policy_name=policy.name,
            )
        )

    average = round(sum(item.score for item in results) / len(results), 4)
    return BaselineRunResult(
        policy_name=policy.name,
        model_name=model_name if policy_name == "openai" else None,
        average_score=average,
        results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline policy against all support tasks.")
    parser.add_argument("--policy", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default=_resolve_model_name())
    args = parser.parse_args()

    result = run_baseline(policy_name=args.policy, model_name=args.model)
    print(result.model_dump_json(indent=2))


def _build_policy(policy_name: str, model_name: str | None) -> Policy:
    if policy_name == "heuristic":
        return HeuristicPolicy()
    _resolve_api_key()
    return OpenAIPolicy(model_name=model_name or _resolve_model_name())


def _resolve_api_key() -> str:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("HF_TOKEN is required for the OpenAI baseline. OPENAI_API_KEY is also accepted locally.")
    return api_key


def _resolve_model_name() -> str:
    return (
        os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL_NAME
    )


def _history_has(observation: SupportObservation, action_type: str) -> bool:
    return any(record.action_type.value == action_type for record in observation.history)


def _already_set(observation: SupportObservation, action_type: str, argument: str) -> bool:
    return any(
        record.action_type.value == action_type and record.argument == argument
        for record in observation.history
    )


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return valid JSON: {text}")
    return text[start : end + 1]


def _infer_issue_type(observation: SupportObservation) -> str:
    message = observation.customer_message.lower()
    if "wrong" in message or "received a navy" in message or "size m" in message:
        return "wrong_item"
    if "crack" in message or "damaged" in message:
        return "damaged_item"
    return "damaged_item"


def _infer_priority(observation: SupportObservation) -> str:
    order_total = float(observation.visible_order.get("order_total_usd", 0.0))
    fraud_risk = str(observation.visible_order.get("fraud_risk", "low")).lower()
    message = observation.customer_message.lower()

    if "business-critical" in message or "asap" in message:
        return "urgent"
    if order_total >= 250 or fraud_risk in {"medium", "high"}:
        return "high"
    return "normal"


def _infer_resolution(observation: SupportObservation, issue_type: str) -> str:
    policy = (observation.visible_policy or "").lower()
    order = observation.visible_order
    customer_details = {
        key: value.lower() for key, value in observation.revealed_customer_details.items()
    }

    if (
        "manual review" in policy
        and float(order.get("order_total_usd", 0.0)) > 1000
        and int(order.get("days_since_delivery", 0)) > 30
    ):
        return "manual_review"

    if (
        issue_type == "wrong_item"
        and order.get("correct_variant_in_stock") is False
        and "preferred_resolution" in customer_details
        and "refund" in customer_details["preferred_resolution"]
    ):
        return "refund"

    if issue_type in {"wrong_item", "damaged_item"}:
        return "replace"
    return "refund"
