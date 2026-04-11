from __future__ import annotations

import os
from typing import Any

from .._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from fastapi import Body, FastAPI, HTTPException
from openenv.core.env_server import HTTPEnvServer, ServerMode
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import ResetResponse, StepResponse

from ..models import (
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
    GraderResult,
    TasksResponse,
)
from .drug_discovery_environment import DrugDiscoveryEnvironment

app = FastAPI(
    title="Drug Discovery OpenEnv",
    description="Lead optimization agent environment (RDKit + OpenEnv).",
    version="1.0.0",
)

openenv_server = HTTPEnvServer(
    DrugDiscoveryEnvironment,
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    max_concurrent_envs=8,
)
openenv_server.register_routes(app, mode=ServerMode.PRODUCTION)

http_env = DrugDiscoveryEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "drug_discovery_env",
        "status": "ok",
        "framework": "openenv",
        "task_count": len(http_env.list_tasks()),
        "endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/tasks",
            "/grader",
            "/docs",
            "/ws",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", response_model=ResetResponse)
def reset_environment(payload: dict[str, Any] = Body(default_factory=dict)) -> ResetResponse:
    task_id = payload.get("task_id")
    seed = payload.get("seed")
    episode_id = payload.get("episode_id")
    try:
        observation = http_env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse)
def step_environment(action: DrugDiscoveryAction) -> StepResponse:
    observation = http_env.step(action)
    return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=DrugDiscoveryState)
def current_state() -> DrugDiscoveryState:
    return http_env.state


@app.get("/grader", response_model=GraderResult)
def grader() -> GraderResult:
    return http_env.grade_current_episode()


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    return TasksResponse(
        tasks=http_env.list_tasks(),
        action_schema=DrugDiscoveryAction.model_json_schema(),
    )


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
