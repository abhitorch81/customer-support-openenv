# Environment design — Drug Discovery OpenEnv

Three-layer OpenEnv pattern:

| Layer | Path |
|-------|------|
| **Models** | `drug_discovery_env/models.py` — `DrugDiscoveryAction`, `DrugDiscoveryObservation`, `DrugDiscoveryState`, grader DTOs |
| **Environment** | `drug_discovery_env/server/drug_discovery_environment.py` — RDKit transforms, rewards, deterministic grader |
| **Server** | `drug_discovery_env/server/app.py` — FastAPI + `HTTPEnvServer`; **`server/app.py`** re-exports for multi-mode |

**Client:** `drug_discovery_env/client.py`

## Use case

Iterative **lead optimization**: agents modify SMILES under **ADMET-style constraints**, observe **descriptor bundles**, and submit with **`stop_and_submit`**. Grading encodes **QED**, **affinity proxy**, **toxicity alerts**, **MW/LogP/TPSA** windows, **workflow** (must `score_molecule` and submit), and optional **scaffold change**.

## Tasks

Defined in `drug_discovery_env/tasks/catalog.json`. Add a row and, if needed, extend `_TASK_ORDER` in `drug_discovery_environment.py`.

## Deployment

- **Dockerfile** — Python 3.11, system libs for RDKit rendering stack, `pip install -r requirements.txt`, `uvicorn server.app:app`.
- **`GET /health`**
- **`openenv.yaml`** — `app: server.app:app`, `port: 7860`
