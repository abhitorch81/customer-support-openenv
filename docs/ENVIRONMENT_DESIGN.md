# Environment design (OpenEnv alignment)

This package follows the **three-component pattern** emphasized in [OpenEnv tutorials](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial) and the [OpenEnv course](https://github.com/raun/openenv-course/tree/main):

| Layer | Role in this repo |
|-------|-------------------|
| **Models** | `customer_support_env/models.py` — typed `SupportAction`, `SupportObservation`, `SupportState`, grader DTOs |
| **Environment** | `customer_support_env/server/customer_support_environment.py` — `Environment` implementation, reward shaping, deterministic grader |
| **Server** | `customer_support_env/server/app.py` — FastAPI + `HTTPEnvServer` (`/ws`), HTTP `/reset` `/step` `/state`, `/health`, `/tasks`, `/grader` |

**Client** (`customer_support_env/client.py`) mirrors reference Hub repos: thin `EnvClient` for remote training or evaluation.

## Use case

**Customer support operations** — agents must gather information under partial observability, follow policy text, and avoid destructive shortcuts (premature resolution, bad close, unnecessary escalation). This is complementary to tool-heavy reference envs ([calendar](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/calendar_env), [reasoning_gym](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/reasoning_gym_env), [repl](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/repl_env)): same OpenEnv contracts, different domain.

## Tasks

Tasks live in `customer_support_env/tasks/catalog.json` and are listed in fixed evaluation order in `SupportTicketEnvironment.list_tasks()`. Adding a task requires:

1. JSON object with `ground_truth` aligned to grader expectations  
2. Entry in the `ordered` list in `list_tasks()`  
3. Optional heuristic updates in `customer_support_env/baseline.py` for smoke-test baselines  

## Deployment

- **Dockerfile** — production image; `PORT` defaults to `7860` for Hugging Face Spaces.  
- **`GET /health`** — `{"status":"healthy"}` for probes and validation scripts.  
- **`openenv.yaml`** — manifest for `openenv` tooling (`app`, `port`, `version`).
