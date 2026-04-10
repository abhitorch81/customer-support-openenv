from __future__ import annotations

from typing import Any

from ._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import MuJoCoAction, MuJoCoObservation, MuJoCoState


class MuJoCoGymEnvClient(EnvClient[MuJoCoAction, MuJoCoObservation, MuJoCoState]):
    """OpenEnv HTTP client for the MuJoCo Gymnasium server."""

    def _step_payload(self, action: MuJoCoAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[MuJoCoObservation]:
        obs_data = payload.get("observation", {})
        observation = MuJoCoObservation.model_validate(
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

    def _parse_state(self, payload: dict[str, Any]) -> MuJoCoState:
        return MuJoCoState.model_validate(payload)
