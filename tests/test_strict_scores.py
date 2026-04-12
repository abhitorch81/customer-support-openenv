"""Hackathon / validator contract: all reported task scores live strictly in (0, 1)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from drug_discovery_env.models import GraderResult
from drug_discovery_env.server.drug_discovery_environment import _strict_unit_interval
from inference import _strict_unit_interval as inference_strict_unit_interval


@pytest.mark.parametrize(
    "raw",
    [0.0, 1.0, 1e-12, 0.999999999, 0.5, -0.1, 1.1, float("nan")],
)
def test_strict_unit_interval_open_bounds(raw: float) -> None:
    y = _strict_unit_interval(raw)
    assert 0.0 < y < 1.0
    assert y == inference_strict_unit_interval(raw)


def test_strict_unit_interval_average_extremes() -> None:
    n = 6
    low = _strict_unit_interval(sum([0.0] * n) / n)
    high = _strict_unit_interval(sum([1.0] * n) / n)
    assert 0.0 < low < 1.0
    assert 0.0 < high < 1.0
    assert low < high


def test_grader_result_rejects_closed_interval_scores() -> None:
    ok_bd = {"a": 0.5, "b": 0.5}
    GraderResult(task_id="t", score=0.5, breakdown=ok_bd, passed=False)
    with pytest.raises(ValidationError):
        GraderResult(task_id="t", score=0.0, breakdown=ok_bd, passed=False)
    with pytest.raises(ValidationError):
        GraderResult(task_id="t", score=1.0, breakdown=ok_bd, passed=False)


def test_grader_result_breakdown_must_be_open_interval() -> None:
    with pytest.raises(ValidationError):
        GraderResult(
            task_id="t",
            score=0.5,
            breakdown={"x": 0.0, "y": 0.5},
            passed=False,
        )
    with pytest.raises(ValidationError):
        GraderResult(
            task_id="t",
            score=0.5,
            breakdown={"x": 1.0, "y": 0.5},
            passed=False,
        )


def test_strict_unit_interval_no_exact_zero_after_rounding() -> None:
    # Values that could round to 0.0 at 6 dp should still map inside (0, 1).
    tiny = 1e-12
    y = _strict_unit_interval(tiny)
    assert y >= 1e-4
    assert not math.isclose(y, 0.0)


def test_strict_unit_interval_no_exact_one_after_rounding() -> None:
    y = _strict_unit_interval(0.999999999999)
    assert y < 1.0
    assert not math.isclose(y, 1.0)
