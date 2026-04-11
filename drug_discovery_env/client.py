from __future__ import annotations

from typing import Any

from ._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState


class DrugDiscoveryEnvClient(EnvClient[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    def _step_payload(self, action: DrugDiscoveryAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[DrugDiscoveryObservation]:
        obs_data = payload.get("observation", {})
        observation = DrugDiscoveryObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward")),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> DrugDiscoveryState:
        return DrugDiscoveryState.model_validate(payload)
