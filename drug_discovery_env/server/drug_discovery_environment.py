from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from .._bootstrap import bootstrap_local_deps

bootstrap_local_deps()

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..catalog import load_task_catalog
from ..chemistry import (
    composite_reward,
    compute_descriptor_bundle,
    format_descriptor_block,
    mol_from_smiles,
    remove_substructure,
    replace_substructure,
    run_canned_reaction,
    scaffold_murcko_smiles,
)
from ..models import (
    ActionType,
    Difficulty,
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
    GraderResult,
    TaskDescriptor,
    TaskMetadata,
)


def _strict_unit_interval(value: float) -> float:
    """Map to strict open interval (0, 1) for Phase 2 validators (reject 0.0 and 1.0)."""
    eps = 1e-4
    v = float(value)
    if v != v:  # NaN
        return eps
    v = min(1.0 - eps, max(eps, v))
    rounded = round(v, 6)
    if rounded <= 0.0 or v <= 0.0:
        return eps
    if rounded >= 1.0 or v >= 1.0:
        return 1.0 - eps
    return float(rounded)


_TASK_ORDER = [
    "lead_optimization_basic",
    "toxicity_aware_optimization",
    "scaffold_hop_challenge",
    "multi_objective_balance",
    "admet_logp_rescue",
]


class DrugDiscoveryEnvironment(Environment[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    """Lead-optimization loop with RDKit-backed transforms and multi-signal grading."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, default_task_id: str = "lead_optimization_basic") -> None:
        super().__init__()
        self._catalog = {t["task_id"]: t for t in load_task_catalog()}
        self._selected_task_id = default_task_id
        self._task: dict[str, Any] | None = None
        self._state: DrugDiscoveryState | None = None
        self._prev_descriptors: dict[str, float] | None = None
        self.reset()

    def select_task(self, task_id: str) -> None:
        if task_id not in self._catalog:
            raise KeyError(f"Unknown task_id: {task_id}")
        self._selected_task_id = task_id

    def list_tasks(self) -> list[TaskDescriptor]:
        out: list[TaskDescriptor] = []
        for tid in _TASK_ORDER:
            t = self._catalog[tid]
            out.append(
                TaskDescriptor(
                    metadata=TaskMetadata(
                        task_id=t["task_id"],
                        difficulty=Difficulty(t["difficulty"]),
                        description=t["description"],
                        objective=t["objective"],
                    ),
                    expected_difficulty_rank=t["difficulty"],
                )
            )
        return out

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> DrugDiscoveryObservation:
        if task_id is not None:
            self.select_task(task_id)
        task = copy.deepcopy(self._catalog[self._selected_task_id])
        self._task = task

        smi = task["seed_smiles"]
        mol = mol_from_smiles(smi)
        if mol is None:
            raise ValueError(f"Invalid seed SMILES for task {task['task_id']}")
        murcko = scaffold_murcko_smiles(mol)

        desc = compute_descriptor_bundle(mol, task["reference_smiles"])
        self._prev_descriptors = None

        seed_score = _strict_unit_interval(0.0)
        self._state = DrugDiscoveryState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task["task_id"],
            difficulty=Difficulty(task["difficulty"]),
            initial_murcko_smiles=murcko,
            smiles=smi,
            done=False,
            cumulative_reward=0.0,
            max_steps=14,
            modification_history=[],
            candidate_pool=[smi],
            best_smiles=smi,
            best_score=seed_score,
            last_descriptors=desc,
            scored_this_episode=False,
            submitted=False,
            invalid_action_count=0,
            current_score=seed_score,
        )
        grader = self.grade_current_episode()
        self._state.current_score = grader.score
        self._state.best_score = grader.score
        return self._build_observation("Environment ready — lead optimization episode started.", 0.0)

    def step(
        self,
        action: DrugDiscoveryAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> DrugDiscoveryObservation:
        state = self._require_state()
        task = self._require_task()
        if state.done:
            return self._build_observation("Episode already finished.", 0.0)

        state.step_count += 1
        reward_components: dict[str, float] = {"time_penalty": -0.02}
        reward_val = -0.02
        outcome = ""

        match action.action_type:
            case ActionType.ADD_GROUP:
                outcome = self._handle_add_group(state, task, action, reward_components)
            case ActionType.REPLACE_SUBSTRUCTURE:
                outcome = self._handle_replace(state, task, action, reward_components)
            case ActionType.REMOVE_GROUP:
                outcome = self._handle_remove(state, task, action, reward_components)
            case ActionType.BIOISOSTERE_SWAP:
                outcome = self._handle_bioisostere(state, task, action, reward_components)
            case ActionType.SCORE_MOLECULE:
                outcome = self._handle_score(state, task, reward_components)
            case ActionType.COMPARE_CANDIDATES:
                outcome = self._handle_compare(state, task, action, reward_components)
            case ActionType.STOP_AND_SUBMIT:
                outcome = self._handle_stop(state, task, reward_components)

        mol = mol_from_smiles(state.smiles)
        if mol is None:
            state.invalid_action_count += 1
            reward_components["invalid_mol"] = -0.5
            reward_val += -0.5
            outcome = "Invalid SMILES state; reset recommended."
        else:
            desc = compute_descriptor_bundle(mol, task["reference_smiles"])
            r_step, _ = composite_reward(desc, task, self._prev_descriptors)
            reward_components["objective"] = r_step
            reward_val += r_step
            self._prev_descriptors = copy.deepcopy(desc)
            state.last_descriptors = desc
            grader = self.grade_current_episode()
            if grader.score > state.best_score:
                state.best_score = grader.score
                state.best_smiles = state.smiles

        state.cumulative_reward = round(state.cumulative_reward + reward_val, 4)
        grader = self.grade_current_episode()
        state.current_score = grader.score

        if state.step_count >= state.max_steps and not state.done:
            state.done = True
            reward_components["max_steps"] = -0.08
            outcome = f"{outcome} Step budget exhausted."

        return self._build_observation(outcome, reward_val)

    @property
    def state(self) -> DrugDiscoveryState:
        s = self._require_state()
        s.current_score = self.grade_current_episode().score
        return s

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="drug_discovery_env",
            description=(
                "Agentic lead optimization OpenEnv: SMILES → RDKit descriptors, "
                "structured medicinal-chemistry actions, multi-objective rewards, "
                "and deterministic grading against target profiles."
            ),
            version="1.0.0",
            author="drug-discovery-openenv",
        )

    def close(self) -> None:
        return None

    def grade_current_episode(self) -> GraderResult:
        state = self._require_state()
        task = self._require_task()
        gt = task["ground_truth"]
        d = state.last_descriptors
        tid = state.task_id

        qed_ok = d["qed"] >= float(gt.get("min_final_qed", 0.2))
        aff_ok = d["affinity_proxy"] >= float(gt.get("min_affinity_proxy", 0.05))
        tox_ok = d["toxicity_proxy"] <= float(gt.get("max_toxicity_proxy", 5))
        mw_ok = d["mw"] <= float(gt.get("max_mw", 600))
        logp_lo = float(gt.get("logp_min", -1))
        logp_hi = float(gt.get("logp_max", 6))
        logp_ok = logp_lo <= d["logp"] <= logp_hi
        logp_final_ok = True
        if "max_final_logp" in gt:
            logp_final_ok = d["logp"] <= float(gt["max_final_logp"])

        scaffold_ok = True
        if gt.get("scaffold_change_required"):
            mol = mol_from_smiles(state.smiles)
            cur = scaffold_murcko_smiles(mol) if mol else ""
            scaffold_ok = cur != state.initial_murcko_smiles and cur != ""

        workflow_ok = True
        if gt.get("require_score_action"):
            workflow_ok = workflow_ok and state.scored_this_episode
        if gt.get("require_submission"):
            workflow_ok = workflow_ok and state.submitted

        checks = {
            "qed": 1.0 if qed_ok else 0.0,
            "affinity": 1.0 if aff_ok else 0.0,
            "toxicity": 1.0 if tox_ok else 0.0,
            "mw_logp": 1.0 if (mw_ok and logp_ok and logp_final_ok) else 0.0,
            "scaffold": 1.0 if scaffold_ok else 0.0,
            "workflow": 1.0 if workflow_ok else 0.0,
        }
        raw = sum(checks.values()) / max(1, len(checks))
        score = _strict_unit_interval(raw)
        breakdown = {k: _strict_unit_interval(v) for k, v in checks.items()}
        passed = score >= 0.72 and qed_ok and aff_ok and tox_ok and mw_ok and logp_ok and workflow_ok
        if gt.get("scaffold_change_required"):
            passed = passed and scaffold_ok
        rationale = (
            f"qed={d['qed']:.2f} aff={d['affinity_proxy']:.3f} tox_alerts={int(d['toxicity_proxy'])} "
            f"MW={d['mw']:.1f} LogP={d['logp']:.2f} submitted={state.submitted} scored={state.scored_this_episode}"
        )
        return GraderResult(
            task_id=tid,
            score=score,
            breakdown=breakdown,
            passed=passed,
            rationale=rationale,
        )

    def _build_observation(self, last_result: str, reward: float) -> DrugDiscoveryObservation:
        state = self._require_state()
        task = self._require_task()
        d = state.last_descriptors
        tp = task["target_profile"]
        target_txt = (
            f"Target: MW≤{tp.get('max_mw')} | LogP {tp.get('logp_min')}–{tp.get('logp_max')} | "
            f"min QED {tp.get('min_qed')} | TPSA≤{tp.get('max_tpsa', 'n/a')}"
        )
        return DrugDiscoveryObservation(
            task_id=state.task_id,
            difficulty=state.difficulty,
            smiles=state.smiles,
            descriptors=copy.deepcopy(d),
            descriptor_text=format_descriptor_block(d),
            target_profile_summary=target_txt,
            modification_history=copy.deepcopy(state.modification_history),
            candidate_pool=copy.deepcopy(state.candidate_pool),
            step_count=state.step_count,
            max_steps=state.max_steps,
            best_score_so_far=state.best_score,
            best_smiles=state.best_smiles,
            available_actions=list(ActionType),
            last_action_result=last_result,
            done=state.done,
            reward=round(reward, 4),
        )

    def _handle_add_group(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        action: DrugDiscoveryAction,
        rc: dict[str, float],
    ) -> str:
        if not action.group_key:
            state.invalid_action_count += 1
            rc["bad_action"] = -0.12
            return "add_group requires group_key (e.g. methyl_aromatic, fluoro_aromatic)."
        new_s = run_canned_reaction(state.smiles, action.group_key)
        if new_s is None:
            rc["failed_edit"] = -0.08
            return f"Canned reaction '{action.group_key}' did not apply."
        state.smiles = new_s
        self._push_candidate(state, new_s)
        state.modification_history.append(f"add_group:{action.group_key}")
        rc["edit_ok"] = 0.12
        return f"Applied add_group {action.group_key}."

    def _handle_replace(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        action: DrugDiscoveryAction,
        rc: dict[str, float],
    ) -> str:
        if not action.query_smarts or not action.replacement_smiles:
            state.invalid_action_count += 1
            rc["bad_action"] = -0.12
            return "replace_substructure requires query_smarts and replacement_smiles."
        new_s = replace_substructure(state.smiles, action.query_smarts, action.replacement_smiles)
        if new_s is None:
            rc["failed_edit"] = -0.1
            return "ReplaceSubstructs produced no valid product."
        state.smiles = new_s
        self._push_candidate(state, new_s)
        state.modification_history.append(f"replace:{action.query_smarts}->{action.replacement_smiles}")
        rc["edit_ok"] = 0.14
        return "Substructure replacement applied."

    def _handle_remove(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        action: DrugDiscoveryAction,
        rc: dict[str, float],
    ) -> str:
        if not action.query_smarts:
            state.invalid_action_count += 1
            rc["bad_action"] = -0.12
            return "remove_group requires query_smarts."
        new_s = remove_substructure(state.smiles, action.query_smarts)
        if new_s is None:
            rc["failed_edit"] = -0.1
            return "RemoveSubstructs failed or not found."
        state.smiles = new_s
        self._push_candidate(state, new_s)
        state.modification_history.append(f"remove:{action.query_smarts}")
        rc["edit_ok"] = 0.12
        return "Substructure removed."

    def _handle_bioisostere(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        action: DrugDiscoveryAction,
        rc: dict[str, float],
    ) -> str:
        key = action.bioisostere_key or "hydroxyl_to_fluoro"
        new_s = run_canned_reaction(state.smiles, key)
        if new_s is None:
            rc["failed_edit"] = -0.08
            return f"Bioisostere '{key}' did not apply."
        state.smiles = new_s
        self._push_candidate(state, new_s)
        state.modification_history.append(f"bioisostere:{key}")
        rc["edit_ok"] = 0.13
        return f"Bioisostere swap {key} applied."

    def _handle_score(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        rc: dict[str, float],
    ) -> str:
        state.scored_this_episode = True
        state.modification_history.append("score_molecule")
        rc["measurement"] = 0.06
        return "Computed/recorded molecular score bundle."

    def _handle_compare(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        action: DrugDiscoveryAction,
        rc: dict[str, float],
    ) -> str:
        pool = state.candidate_pool
        if len(pool) < 2:
            rc["compare_fail"] = -0.05
            return "Need at least two candidates in pool to compare."
        idxs = action.candidate_indices or [0, 1]
        best_i = idxs[0]
        best_s = pool[best_i] if 0 <= best_i < len(pool) else pool[0]
        best_score = -1e9
        for i in idxs:
            if 0 <= i < len(pool):
                m = mol_from_smiles(pool[i])
                if m:
                    d = compute_descriptor_bundle(m, task["reference_smiles"])
                    r, _ = composite_reward(d, task, None)
                    if r > best_score:
                        best_score = r
                        best_s = pool[i]
        state.smiles = best_s
        state.modification_history.append(f"compare:selected={best_s}")
        rc["compare_ok"] = 0.08
        return f"Selected candidate SMILES after comparison."

    def _handle_stop(
        self,
        state: DrugDiscoveryState,
        task: dict[str, Any],
        rc: dict[str, float],
    ) -> str:
        state.submitted = True
        state.done = True
        rc["submit"] = 0.18
        return "stop_and_submit: final molecule locked for grading."

    @staticmethod
    def _push_candidate(state: DrugDiscoveryState, smi: str) -> None:
        if smi not in state.candidate_pool:
            state.candidate_pool.append(smi)
        state.candidate_pool = state.candidate_pool[-5:]

    def _require_state(self) -> DrugDiscoveryState:
        if self._state is None:
            raise RuntimeError("State not initialized.")
        return self._state

    def _require_task(self) -> dict[str, Any]:
        if self._task is None:
            raise RuntimeError("Task not initialized.")
        return self._task
