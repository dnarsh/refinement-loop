"""Acceptance tests."""

from __future__ import annotations

import pytest

from refinement_loop.acceptance import (
    AcceptanceContext,
    BudgetAcceptance,
    ThresholdAcceptance,
)


def _ctx(score=None, latency=0.0, round_index=1, max_rounds=3) -> AcceptanceContext:
    return AcceptanceContext(
        round_index=round_index,
        max_rounds=max_rounds,
        judge_score=score,
        cumulative_latency_sec=latency,
    )


def test_threshold_accepts_when_score_meets() -> None:
    a = ThresholdAcceptance(threshold=4)
    assert a.accept(_ctx(score=4))
    assert a.accept(_ctx(score=5))
    assert not a.accept(_ctx(score=3))


def test_threshold_no_judge_default_accepts() -> None:
    a = ThresholdAcceptance(threshold=4)
    assert a.accept(_ctx(score=None))


def test_threshold_no_judge_can_reject() -> None:
    a = ThresholdAcceptance(threshold=4, accept_when_no_judge=False)
    assert not a.accept(_ctx(score=None))


def test_threshold_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        ThresholdAcceptance(threshold=0)
    with pytest.raises(ValueError):
        ThresholdAcceptance(threshold=6)


def test_budget_accepts_on_score_meet() -> None:
    a = BudgetAcceptance(threshold=4, max_seconds=30.0)
    assert a.accept(_ctx(score=4, latency=1.0))


def test_budget_accepts_when_over_budget_even_low_score() -> None:
    a = BudgetAcceptance(threshold=4, max_seconds=10.0)
    assert a.accept(_ctx(score=2, latency=11.0))


def test_budget_rejects_when_low_score_under_budget() -> None:
    a = BudgetAcceptance(threshold=4, max_seconds=30.0)
    assert not a.accept(_ctx(score=2, latency=1.0))


def test_budget_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        BudgetAcceptance(threshold=0)
    with pytest.raises(ValueError):
        BudgetAcceptance(max_seconds=0)
