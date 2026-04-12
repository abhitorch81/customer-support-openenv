"""Smoke and integration checks for DrugDiscoveryEnvironment."""

from __future__ import annotations

import pytest

from drug_discovery_env.environment import DrugDiscoveryEnvironment
from drug_discovery_env.models import ActionType, DrugDiscoveryAction


def _assert_open_unit_score(env: DrugDiscoveryEnvironment) -> None:
    g = env.grade_current_episode()
    assert g.task_id
    assert 0.0 < g.score < 1.0, f"score out of open interval: {g.score!r}"
    assert g.breakdown
    for k, v in g.breakdown.items():
        assert 0.0 < float(v) < 1.0, f"breakdown[{k}]={v!r}"


def test_list_tasks_has_five_ordered_entries() -> None:
    env = DrugDiscoveryEnvironment()
    tasks = env.list_tasks()
    assert len(tasks) == 5
    ids = [t.metadata.task_id for t in tasks]
    assert ids[0] == "lead_optimization_basic"
    assert "admet_logp_rescue" in ids


@pytest.mark.parametrize(
    "task_id",
    [
        "lead_optimization_basic",
        "toxicity_aware_optimization",
        "scaffold_hop_challenge",
        "multi_objective_balance",
        "admet_logp_rescue",
    ],
)
def test_grader_score_strict_open_interval_per_task(task_id: str) -> None:
    env = DrugDiscoveryEnvironment()
    env.reset(task_id=task_id)
    _assert_open_unit_score(env)


def test_state_scores_strict_open_interval_after_reset() -> None:
    env = DrugDiscoveryEnvironment()
    env.reset(task_id="lead_optimization_basic")
    s = env.state
    assert 0.0 < s.best_score < 1.0
    assert 0.0 < s.current_score < 1.0


def test_step_score_molecule_then_grader_stays_valid() -> None:
    env = DrugDiscoveryEnvironment()
    env.reset(task_id="lead_optimization_basic")
    obs = env.step(DrugDiscoveryAction(action_type=ActionType.SCORE_MOLECULE))
    assert obs.step_count >= 1
    _assert_open_unit_score(env)


def test_stop_and_submit_marks_done() -> None:
    env = DrugDiscoveryEnvironment()
    env.reset(task_id="lead_optimization_basic")
    env.step(DrugDiscoveryAction(action_type=ActionType.SCORE_MOLECULE))
    obs = env.step(DrugDiscoveryAction(action_type=ActionType.STOP_AND_SUBMIT))
    assert obs.done is True
    _assert_open_unit_score(env)


def test_unknown_task_raises() -> None:
    env = DrugDiscoveryEnvironment()
    with pytest.raises(KeyError):
        env.reset(task_id="nonexistent_task_xyz")
