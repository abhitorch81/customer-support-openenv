from __future__ import annotations

import os
from typing import Any

from .._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from fastapi import Body, FastAPI, HTTPException
from openenv.core.env_server import HTTPEnvServer, ServerMode
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import ResetResponse, StepResponse

from ..models import GraderResult, MuJoCoAction, MuJoCoObservation, MuJoCoState, TasksResponse
from .mujoco_gym_environment import MuJoCoGymEnvironment

app = FastAPI(
    title="MuJoCo Gym OpenEnv",
    description="Gymnasium MuJoCo environment behind the OpenEnv HTTP/WebSocket server.",
    version="0.1.0",
)

openenv_server = HTTPEnvServer(
    MuJoCoGymEnvironment,
    MuJoCoAction,
    MuJoCoObservation,
    max_concurrent_envs=4,
)
openenv_server.register_routes(app, mode=ServerMode.PRODUCTION)

http_env = MuJoCoGymEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "mujoco_gym_env",
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
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse)
def step_environment(action: MuJoCoAction) -> StepResponse:
    try:
        observation = http_env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=MuJoCoState)
def current_state() -> MuJoCoState:
    return http_env.state


@app.get("/grader", response_model=GraderResult)
def grader() -> GraderResult:
    return http_env.grade_current_episode()


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    return TasksResponse(
        tasks=http_env.list_tasks(),
        action_schema=MuJoCoAction.model_json_schema(),
    )


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
