# Environment design (OpenEnv + MuJoCo)

This package follows the **three-component pattern** from [OpenEnv tutorials](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial) and the [OpenEnv course](https://github.com/raun/openenv-course/tree/main):

| Layer | Role |
|-------|------|
| **Models** | `mujoco_gym_env/models.py` — `MuJoCoAction`, `MuJoCoObservation`, `MuJoCoState`, task + grader DTOs |
| **Environment** | `mujoco_gym_env/server/mujoco_gym_environment.py` — `Environment` over Gymnasium MuJoCo v5 |
| **Server** | `mujoco_gym_env/server/app.py` — FastAPI + `HTTPEnvServer` (`/ws`), `/reset`, `/step`, `/state`, `/health`, `/tasks`, `/grader` |

**Client** (`mujoco_gym_env/client.py`): thin `EnvClient` for remote rollouts.

## Use case

**Continuous control under physics**: agents map vectors to torques, receive dense rewards from MuJoCo, and are scored with a **deterministic** rubric (length + return + failure/invalid penalties). Suited for RL, control, and robustness evaluation—orthogonal to tool-use or ticket workflows.

## Tasks

Tasks are registered in `mujoco_gym_env/server/mujoco_gym_environment.py` (`_TASK_REGISTRY`, `_TASK_ORDER`). Adding a task:

1. Append a registry entry (`gym_id`, copy, `difficulty`, `rank`, `return_offset`, `return_scale` for the grader).
2. Add its id to `_TASK_ORDER`.

## Deployment

- **Dockerfile** — `python:3.11-slim-bookworm`, `pip install -r requirements.txt`, `uvicorn server.app:app`, `PORT` default `7860` for Hugging Face Spaces.
- **`GET /health`** — `{"status":"healthy"}`.
- **`openenv.yaml`** — `app: server.app:app`, `port: 7860` (multi-mode tooling); implementation in `mujoco_gym_env/server/app.py`.
