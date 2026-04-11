from __future__ import annotations

from .environment import DrugDiscoveryEnvironment
from .models import ActionType, DrugDiscoveryAction, DrugDiscoveryObservation


class HeuristicPolicy:
    """Rule-based policy for smoke tests and inference fallback."""

    name = "heuristic"

    def act(self, observation: DrugDiscoveryObservation) -> DrugDiscoveryAction:
        tid = observation.task_id
        hist = observation.modification_history

        if not any(h == "score_molecule" for h in hist):
            return DrugDiscoveryAction(action_type=ActionType.SCORE_MOLECULE)

        if tid == "toxicity_aware_optimization" and not any("nitro_to_amino" in h for h in hist):
            return DrugDiscoveryAction(
                action_type=ActionType.BIOISOSTERE_SWAP,
                bioisostere_key="nitro_to_amino",
            )

        if tid == "admet_logp_rescue" and not any("remove:" in h for h in hist):
            return DrugDiscoveryAction(
                action_type=ActionType.REMOVE_GROUP,
                query_smarts="CCCCCC",
            )

        if tid == "scaffold_hop_challenge" and not any("fluoro_aromatic" in h for h in hist):
            return DrugDiscoveryAction(
                action_type=ActionType.ADD_GROUP,
                group_key="fluoro_aromatic",
            )

        if not any("replace:[OH]" in h for h in hist):
            return DrugDiscoveryAction(
                action_type=ActionType.REPLACE_SUBSTRUCTURE,
                query_smarts="[OH]",
                replacement_smiles="F",
            )

        if not any("add_group:methyl_aromatic" in h for h in hist) and observation.step_count < 10:
            return DrugDiscoveryAction(
                action_type=ActionType.ADD_GROUP,
                group_key="methyl_aromatic",
            )

        return DrugDiscoveryAction(action_type=ActionType.STOP_AND_SUBMIT)


def main() -> None:
    env = DrugDiscoveryEnvironment()
    pol = HeuristicPolicy()
    try:
        for task in env.list_tasks():
            tid = task.metadata.task_id
            env.select_task(tid)
            obs = env.reset()
            while not obs.done:
                obs = env.step(pol.act(obs))
            g = env.grade_current_episode()
            print(f"task={tid} score={g.score:.4f} passed={g.passed} {g.rationale}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
