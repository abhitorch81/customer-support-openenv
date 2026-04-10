---
title: MuJoCo Gym OpenEnv
emoji: "🦎"
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
short_description: "OpenEnv MuJoCo Gymnasium v5 — 3 physics control tasks"
---

# MuJoCo Gym OpenEnv

**OpenEnv** environment for **MuJoCo physics** through **Gymnasium v5**: real continuous control, dense rewards, and a **deterministic grader**—a strong complement to text-only or calendar-style envs.

## Why this is a strong evaluation surface

- **Real physics**: MuJoCo dynamics, contact, and torque limits—not a hand-written state machine.
- **Partial observability only through the simulator**: agents consume **vector observations** and emit **boxed actions** (no hidden “oracle” shortcuts).
- **Multi-task ramp**: **easy → medium → hard** (`InvertedPendulum-v5`, `Hopper-v5`, `HalfCheetah-v5`).
- **Same deployment pattern as reference OpenEnv repos**: Pydantic **Action / Observation / State**, `Environment` implementation, **FastAPI + `HTTPEnvServer`**, **`/health`**, WebSocket **`/ws`**, HTTP **`/reset`**, **`/step`**, **`/state`**, **`/tasks`**, **`/grader`**.

## Tasks (fixed evaluation order)

| `task_id` | Gymnasium env | Rank |
|-----------|---------------|------|
| `inverted_pendulum_v5` | `InvertedPendulum-v5` | easy |
| `hopper_v5` | `Hopper-v5` | medium |
| `halfcheetah_v5` | `HalfCheetah-v5` | hard |

## Action & observation

- **Action** (`MuJoCoAction`): `control: list[float]` with length = environment `action_dim` (invalid length → penalty, episode continues).
- **Observation** (`MuJoCoObservation`): `obs` (flattened float vector), `action_dim`, `obs_dim`, `max_episode_steps`, `step_count`, `reward`, `done`, plus `last_info` from Gymnasium.

## Quick start

```powershell
cd D:\path\to\mujoco-gym-openenv
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### API server

```powershell
.venv\Scripts\python.exe -m mujoco_gym_env.server
```

- Docs: `http://127.0.0.1:8000/docs` (or `PORT` / `7860` on Spaces)
- Health: `GET /health`

### Baseline (random policy, one episode per task)

```powershell
.venv\Scripts\python.exe -m mujoco_gym_env.baseline
```

### Lockfile (multi-mode / `uv`)

```powershell
uv lock --python 3.11
```

Commit the generated **`uv.lock`** so OpenEnv multi-mode checks pass.

### Validator-style `inference.py`

Matches the **hackathon inference spec**:

| Variable | Role |
|----------|------|
| `API_BASE_URL` | LLM endpoint (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model id (default: `openai/gpt-4.1-mini`) |
| `HF_TOKEN` | **Preferred** API key for the proxy (same as Hugging Face token) |
| `API_KEY` | Alternative if `HF_TOKEN` is unset |
| `BENCHMARK` | Logged as `env=` in `[START]` (default: `mujoco_gym_env`) |
| `LOCAL_IMAGE_NAME` / `IMAGE_NAME` | For `from_docker_image()` flows only; read for compatibility, unused in-process |

- **OpenAI client:** `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or API_KEY)`.
- **LLM traffic:** each task does a minimal **`chat.completions`** (fallback **`responses`**) so LiteLLM sees your key; MuJoCo torques stay **random** for stability.
- **Stdout:** only `[START]` → one `[STEP]` per `env.step()` → `[END]` after `env.close()` (always, including on errors). Rewards **`%.2f`**; end score **`%.2f`** in `[0, 1]`.

```powershell
$env:HF_TOKEN="..."; .venv\Scripts\python.exe inference.py
```

Local validation (optional): Docker build + `openenv validate` as in the official `validate-submission.sh` pattern (Dockerfile at **repo root**).

## OpenEnv layout

| Layer | Path |
|--------|------|
| Models | [mujoco_gym_env/models.py](mujoco_gym_env/models.py) |
| Environment | [mujoco_gym_env/server/mujoco_gym_environment.py](mujoco_gym_env/server/mujoco_gym_environment.py) |
| Server | [mujoco_gym_env/server/app.py](mujoco_gym_env/server/app.py); HF / multi-mode entry [server/app.py](server/app.py) |
| Client | [mujoco_gym_env/client.py](mujoco_gym_env/client.py) |
| Manifest | [openenv.yaml](openenv.yaml) |

## Project structure

```text
├── mujoco_gym_env/
│   ├── baseline.py
│   ├── client.py
│   ├── environment.py
│   ├── models.py
│   └── server/
│       ├── app.py
│       └── mujoco_gym_environment.py
├── server/
│   ├── __init__.py
│   └── app.py
├── scripts/
│   ├── run_baseline.py
│   └── smoke_mujoco_gym.py
├── docs/
│   └── ENVIRONMENT_DESIGN.md
├── Dockerfile
├── openenv.yaml
├── inference.py
├── pyproject.toml
└── requirements.txt
```

## Learning resources

- [OpenEnv tutorials](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)
- [Building RL Environments with OpenEnv](https://github.com/raun/openenv-course/tree/main)
- [Gymnasium MuJoCo envs](https://gymnasium.farama.org/environments/mujoco/)

## Grader

The grader combines **episode length progress**, **normalized return** (per-task scale), **early-failure penalty** (terminate without truncation), and **invalid-action** penalties. Scores are clamped to a strict **(0, 1)** interval for parsers that reject `0.0` / `1.0` endpoints.

---

*PyPI / install name: `mujoco-gym-openenv` (`pip install -e .` from this repo).*
