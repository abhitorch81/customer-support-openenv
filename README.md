---
title: Customer Support OpenEnv
emoji: "📬"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: OpenEnv customer support ticket environment
---

# Customer Support OpenEnv

Customer Support OpenEnv is an official OpenEnv environment for training and evaluating agents on a real customer operations workflow: deciding how to resolve e-commerce support tickets.

The environment simulates the kind of work human support agents do every day:

- inspect order details
- inspect refund and replacement policy
- ask for missing customer information
- classify the issue
- set priority
- decide the right resolution
- escalate when policy exceptions or risk flags require human review

This domain scores well for the OpenEnv hackathon because it is real-world, deterministic to grade, and supports meaningful partial rewards instead of only binary success.

## Why this task

This environment models a genuine operational workflow rather than a toy task. It is useful for agent evaluation because strong performance requires:

- targeted information gathering instead of random action spam
- policy-aware decision making
- handling missing information correctly
- knowing when to escalate instead of acting too early

## Tasks

The environment ships with 3 deterministic tasks:

1. `easy_damaged_mug`
   Difficulty: easy
   Objective: replace a clearly damaged order with enough information already available.

2. `medium_wrong_hoodie`
   Difficulty: medium
   Objective: resolve a wrong-item ticket, but only after collecting verification and choosing refund vs replacement based on stock and customer preference.

3. `hard_laptop_policy_exception`
   Difficulty: hard
   Objective: handle a damaged high-value order outside the normal policy window that requires evidence collection, exception escalation, and disciplined manual-review handling.

## Action space

Actions are typed JSON objects with the following schema:

```json
{
  "action_type": "lookup_order | lookup_policy | ask_customer | set_issue_type | set_priority | decide_resolution | escalate_ticket | close_ticket",
  "argument": "optional string payload",
  "notes": "optional free-form reasoning note"
}
```

Common examples:

```json
{"action_type": "lookup_order"}
{"action_type": "lookup_policy"}
{"action_type": "ask_customer", "argument": "photo_of_received_item"}
{"action_type": "set_issue_type", "argument": "wrong_item"}
{"action_type": "set_priority", "argument": "urgent"}
{"action_type": "decide_resolution", "argument": "manual_review"}
{"action_type": "escalate_ticket"}
{"action_type": "close_ticket"}
```

## Observation space

Each observation is a typed JSON object that includes:

- task metadata
- customer message
- the current visible order snapshot
- visible policy text if opened
- revealed customer details from prior questions
- action history
- last action result
- step count and reward values

The observation only shows information the agent has earned or revealed so far, which makes information gathering meaningful.

## Reward design

The reward function provides dense trajectory feedback:

- positive reward for retrieving the order and policy
- positive reward for asking for truly missing required information
- positive reward for correct issue classification
- positive reward for correct priority assignment
- strong positive reward for the correct final resolution
- positive reward for correct escalation behavior
- small time penalty every step
- penalties for repeated, irrelevant, premature, or destructive actions

This gives agents signal throughout the episode while still preserving a final deterministic grader score.

## OpenEnv framework mapping

This repo follows the official OpenEnv creator pattern (see also [OpenEnv tutorials](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)):

- [customer_support_env/models.py](customer_support_env/models.py): typed `Action`, `Observation`, and `State` models built on OpenEnv base types
- [customer_support_env/client.py](customer_support_env/client.py): official `EnvClient` implementation
- [customer_support_env/server/customer_support_environment.py](customer_support_env/server/customer_support_environment.py): `Environment` implementation
- [customer_support_env/server/app.py](customer_support_env/server/app.py): FastAPI app with OpenEnv WebSocket server plus hackathon HTTP endpoints
- [openenv.yaml](openenv.yaml): OpenEnv manifest (port aligned with Docker Space: `7860`)

## Project structure

```text
customer_support_openenv/
├── customer_support_env/
│   ├── baseline.py
│   ├── catalog.py
│   ├── client.py
│   ├── environment.py
│   ├── models.py
│   ├── server/
│   │   ├── __main__.py
│   │   ├── app.py
│   │   └── customer_support_environment.py
│   └── tasks/
│       └── catalog.json
├── scripts/
│   └── run_baseline.py
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
└── requirements.txt
```

## Local setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
cd D:\path\to\customer-support-openenv
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install dependencies

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If PowerShell script activation is enabled on your machine, you can also activate the venv normally. Otherwise, just keep using `.venv\Scripts\python.exe`.

If you want to try the official validator:

```powershell
.venv\Scripts\python.exe -m pip install openenv-core
openenv validate
```

## Run the environment

### Start the API server

```powershell
.venv\Scripts\python.exe -m customer_support_env.server
```

The server starts at:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`
- OpenEnv WebSocket endpoint: `ws://127.0.0.1:8000/ws`

### Quick manual test

Reset to the easy task:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/reset -ContentType 'application/json' -Body '{"task_id":"easy_damaged_mug"}'
```

The reset response now follows the official OpenEnv HTTP response shape:

```json
{
  "observation": { "...": "..." },
  "reward": 0.0,
  "done": false
}
```

Take one step:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/step -ContentType 'application/json' -Body '{"action_type":"lookup_order"}'
```

List tasks:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/tasks
```

Get grader score:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/grader
```

### Official OpenEnv client usage

```python
from customer_support_env import CustomerSupportEnv, SupportAction

with CustomerSupportEnv(base_url="http://127.0.0.1:8000").sync() as env:
    result = env.reset(task_id="easy_damaged_mug")
    print(result.observation.task_id)

    result = env.step(SupportAction(action_type="lookup_order"))
    print(result.observation.visible_order["order_id"])
```

## Run the baselines

### Heuristic baseline

This is the fastest local smoke test and does not require an API key. It is intentionally generic and should not be expected to solve every harder case perfectly.

```powershell
.venv\Scripts\python.exe scripts\run_baseline.py --policy heuristic
```

### OpenAI baseline

The hackathon submission uses the required environment variables:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="openai/gpt-4.1-mini"
$env:HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

Then run:

```powershell
.venv\Scripts\python.exe inference.py
```

For local experimentation, the older entrypoint still works too:

```powershell
.venv\Scripts\python.exe scripts\run_baseline.py --policy openai
```

Backwards compatibility note:
- `HF_TOKEN` is the primary submission credential
- `MODEL_NAME` is the primary submission model variable
- `OPENAI_API_KEY` and `OPENAI_MODEL` are also accepted locally

## Docker

Build:

```powershell
docker build -t customer-support-openenv .
```

Run:

```powershell
docker run -p 7860:7860 customer-support-openenv
```

Then open:

- `http://127.0.0.1:7860/docs`

## Hugging Face Spaces

Recommended Space type:

- Docker

Recommended metadata:

- tag the Space with `openenv`

Suggested startup:

- use the provided `Dockerfile`
- keep port `7860`

## Baseline expectations

You should treat the heuristic baseline as a smoke test and the OpenAI baseline as the reproducible submission baseline.

Current verified heuristic baseline:

- `easy_damaged_mug`: `1.00`
- `medium_wrong_hoodie`: `1.00`
- `hard_laptop_policy_exception`: `0.8728`
- average: `0.9576`

To regenerate:

```powershell
.venv\Scripts\python.exe scripts\run_baseline.py --policy heuristic
.venv\Scripts\python.exe inference.py
```

The hard task can still reach `1.0` with the correct gold path, but the heuristic baseline intentionally underperforms there to preserve task difficulty.
