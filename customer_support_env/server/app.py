from __future__ import annotations

import os
from typing import Any

from .._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from fastapi import Body, FastAPI, HTTPException, Query
from openenv.core.env_server import HTTPEnvServer, ServerMode
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import ResetResponse, StepResponse

from ..baseline import run_baseline
from ..models import GraderResult, SupportAction, SupportObservation, SupportState, TasksResponse
from .customer_support_environment import SupportTicketEnvironment

app = FastAPI(
    title="Customer Support OpenEnv",
    description="Official OpenEnv customer support ticket resolution environment.",
    version="0.2.0",
)

openenv_server = HTTPEnvServer(
    SupportTicketEnvironment,
    SupportAction,
    SupportObservation,
    max_concurrent_envs=8,
)
openenv_server.register_routes(app, mode=ServerMode.PRODUCTION)

http_env = SupportTicketEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "customer_support_env",
        "status": "ok",
        "framework": "openenv",
        "task_count": len(http_env.list_tasks()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/ws"],
    }


@app.post("/reset", response_model=ResetResponse)
def reset_environment(payload: dict[str, Any] = Body(default_factory=dict)) -> ResetResponse:
    task_id = payload.get("task_id")
    seed = payload.get("seed")
    episode_id = payload.get("episode_id")
    try:
        observation = http_env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse)
def step_environment(action: SupportAction) -> StepResponse:
    observation = http_env.step(action)
    return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=SupportState)
def current_state() -> SupportState:
    return http_env.state


@app.get("/grader", response_model=GraderResult)
def grader() -> GraderResult:
    return http_env.grade_current_episode()


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    return TasksResponse(
        tasks=http_env.list_tasks(),
        action_schema=SupportAction.model_json_schema(),
    )


@app.get("/baseline")
def baseline(
    policy: str = Query(default="heuristic", pattern="^(heuristic|openai)$"),
    model: str = Query(default=os.environ.get("MODEL_NAME", "openai/gpt-4.1-mini")),
) -> dict[str, object]:
    try:
        result = run_baseline(policy_name=policy, model_name=model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
