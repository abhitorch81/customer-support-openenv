---
title: Drug Discovery OpenEnv
emoji: "🧬"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
short_description: "OpenEnv: RDKit lead optimization, 5 tasks"
---

# Drug Discovery OpenEnv — Lead Optimization Agent

**OpenEnv** environment for **agentic medicinal chemistry**: start from a **seed SMILES**, apply **structured edits**, read **RDKit descriptors**, and optimize toward a **target profile** with a **deterministic grader**.

## Why this case study

- **Real workflow**: lead optimization loops (edit → measure → tradeoffs), not toy chat.
- **Tool-like actions**: typed JSON actions (not free-text-only agents).
- **Measurable science**: MW, LogP, HBD/HBA, TPSA, **QED**, **SA score**, **PAINS-like alerts**, **affinity proxy** (Morgan Tanimoto to a reference ligand).
- **Multi-task**: five episodes (lead opt, toxicity-aware, scaffold hop, multi-objective, ADMET rescue).

## Tasks (fixed order)

1. `lead_optimization_basic` — easy  
2. `toxicity_aware_optimization` — medium  
3. `scaffold_hop_challenge` — medium (Murcko scaffold change required to pass)  
4. `multi_objective_balance` — hard  
5. `admet_logp_rescue` — hard (lipophilic rescue)

## Actions (`DrugDiscoveryAction`)

| `action_type` | Role |
|---------------|------|
| `add_group` | Canned reaction (`group_key`: `methyl_aromatic`, `fluoro_aromatic`, …) |
| `replace_substructure` | RDKit `ReplaceSubstructs` (`query_smarts`, `replacement_smiles`) |
| `remove_group` | `DeleteSubstructs` (`query_smarts`) |
| `bioisostere_swap` | Canned map (`bioisostere_key`: `hydroxyl_to_fluoro`, `nitro_to_amino`, …) |
| `score_molecule` | Registers measurement (required for grader workflow) |
| `compare_candidates` | Picks among `candidate_pool` by composite proxy |
| `stop_and_submit` | Ends episode (required for full pass) |

## Reward (step)

Weighted mix of **affinity proxy**, **QED**, **toxicity alerts**, **SA**, and **constraint violations** vs task `target_profile`, plus deltas vs previous step.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m drug_discovery_env.server
```

- Docs: `http://127.0.0.1:8000/docs`  
- Health: `GET /health`

```powershell
.venv\Scripts\python -m drug_discovery_env.baseline
```

## `inference.py` (hackathon)

- `API_BASE_URL`, `MODEL_NAME`, **`HF_TOKEN` or `API_KEY`**, optional `BENCHMARK`, `LOCAL_IMAGE_NAME` / `IMAGE_NAME`.
- **OpenAI** client hits the proxy once per task; rollouts use the **heuristic** policy (stable SMILES pipeline).
- Stdout: **`[START]`** / **`[STEP]`** / **`[END]`** only.

```powershell
$env:HF_TOKEN="..."; .venv\Scripts\python inference.py
```

## Layout

| Layer | Path |
|--------|------|
| Models | `drug_discovery_env/models.py` |
| RDKit engine | `drug_discovery_env/chemistry.py` |
| Environment | `drug_discovery_env/server/drug_discovery_environment.py` |
| Server | `drug_discovery_env/server/app.py` + **`server/app.py`** (multi-mode) |
| Tasks | `drug_discovery_env/tasks/catalog.json` |

## Lockfile

```powershell
uv lock --python 3.11
```

Commit **`uv.lock`** for multi-mode checks.

## References

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [RDKit](https://www.rdkit.org/)
