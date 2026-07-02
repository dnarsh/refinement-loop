"""RefinementLoop integration tests.

Uses controllable mock Engine + Judge so the loop's branching is
deterministic across runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from refinement_loop import (
    BudgetAcceptance,
    LoopResult,
    RefinementLoop,
    StubReformulator,
    ThresholdAcceptance,
)


# ── Test doubles (real, not mocks) ───────────────────────────


@dataclass
class FakeAnswer:
    text: str
    citations: List[str]
    metadata: Dict[str, Any]


class FakeEngine:
    """Records each query it received; returns canned answers based on
    a callable that maps round_index → answer text. The round index is
    determined by how many times run() has been called so far."""

    def __init__(self, name: str, response_by_round) -> None:
        self.name = name
        self.queries: List[str] = []
        self._response_by_round = response_by_round

    def run(self, query: str) -> FakeAnswer:
        self.queries.append(query)
        text = self._response_by_round(len(self.queries))
        return FakeAnswer(text=text, citations=[], metadata={})


@dataclass
class FakeJudgeResult:
    score: int
    reasoning: str
    judge_name: str = "fake"


class FakeJudge:
    """Returns judge scores from a list, one per call."""

    def __init__(self, name: str, scores: List[int]) -> None:
        self.name = name
        self._scores = list(scores)
        self.calls = 0

    def judge(self, case, answer) -> FakeJudgeResult:
        score = (
            self._scores[self.calls]
            if self.calls < len(self._scores)
            else self._scores[-1]
        )
        self.calls += 1
        return FakeJudgeResult(score=score, reasoning=f"round {self.calls}")


# ── Tests ────────────────────────────────────────────────────


def test_loop_stops_when_first_round_meets_threshold() -> None:
    engine = FakeEngine("e", lambda r: f"answer round {r}")
    judge = FakeJudge("j", scores=[5])  # first round is a 5 — accept immediately

    loop = RefinementLoop(
        engine=engine,
        judge=judge,
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=3,
    )
    result: LoopResult = loop.run("What is X?")

    assert result.total_rounds == 1
    assert result.final_score == 5
    assert result.final_answer_text == "answer round 1"
    assert engine.queries == ["What is X?"]
    assert result.final_round.accepted is True


def test_loop_refines_then_accepts_on_better_round() -> None:
    engine = FakeEngine("e", lambda r: f"answer round {r}")
    judge = FakeJudge("j", scores=[2, 5])  # round 1 fails, round 2 passes

    loop = RefinementLoop(
        engine=engine,
        judge=judge,
        reformulator=StubReformulator(),
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=3,
    )
    result = loop.run("What is X?")

    assert result.total_rounds == 2
    assert result.final_score == 5
    assert result.rounds[0].accepted is False
    assert result.rounds[1].accepted is True
    # Round 2 query should be reformulated (≠ original)
    assert engine.queries[0] == "What is X?"
    assert engine.queries[1] != "What is X?"
    assert "Be specific" in engine.queries[1]


def test_loop_respects_max_rounds_when_nothing_meets_threshold() -> None:
    engine = FakeEngine("e", lambda r: "low quality")
    judge = FakeJudge("j", scores=[1, 1, 1, 1])  # always fail

    loop = RefinementLoop(
        engine=engine,
        judge=judge,
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=3,
    )
    result = loop.run("What is X?")

    assert result.total_rounds == 3, "must cap at max_rounds"
    assert result.final_score == 1
    assert result.final_round.accepted is False
    assert "max_rounds=3 reached" in result.final_round.accept_reason


def test_loop_with_no_judge_runs_exactly_once() -> None:
    """No judge = no scoring possible = no refinement; accept after 1 round."""
    engine = FakeEngine("e", lambda r: "answer")
    loop = RefinementLoop(engine=engine, judge=None, max_rounds=5)
    result = loop.run("query")

    assert result.total_rounds == 1
    assert result.final_score is None
    assert engine.queries == ["query"]


def test_loop_rejects_empty_query() -> None:
    loop = RefinementLoop(engine=FakeEngine("e", lambda r: ""))
    with pytest.raises(ValueError, match="non-empty"):
        loop.run("")
    with pytest.raises(ValueError, match="non-empty"):
        loop.run("   ")


def test_loop_rejects_bad_max_rounds() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        RefinementLoop(engine=FakeEngine("e", lambda r: ""), max_rounds=0)


def test_loop_records_per_round_trace() -> None:
    engine = FakeEngine("e", lambda r: f"text {r}")
    judge = FakeJudge("j", scores=[2, 3, 5])

    loop = RefinementLoop(
        engine=engine,
        judge=judge,
        acceptance=ThresholdAcceptance(threshold=5),
        max_rounds=3,
    )
    result = loop.run("q")

    assert [r.round_index for r in result.rounds] == [1, 2, 3]
    assert [r.judge_score for r in result.rounds] == [2, 3, 5]
    assert [r.answer_text for r in result.rounds] == ["text 1", "text 2", "text 3"]
    assert result.rounds[-1].accepted is True


def test_loop_with_budget_acceptance_stops_on_budget() -> None:
    """A slow engine (100ms/call) with a tight budget (50ms) trips the
    budget after the first round even though the judge would have
    rejected the answer.

    Uses 100ms/50ms intervals rather than 1ms scales because Windows
    time.monotonic() resolution can be as coarse as 15-16ms.
    """
    import time

    class SlowEngine:
        name = "slow"

        def run(self, query: str) -> FakeAnswer:
            time.sleep(0.1)  # 100ms per call — comfortably above clock resolution
            return FakeAnswer(text="answer", citations=[], metadata={})

    judge = FakeJudge("j", scores=[1, 1, 1])
    loop = RefinementLoop(
        engine=SlowEngine(),
        judge=judge,
        acceptance=BudgetAcceptance(threshold=5, max_seconds=0.05),
        max_rounds=3,
    )
    result = loop.run("q")

    # 100ms >> 50ms budget → budget trips after round 1
    assert result.total_rounds == 1
    assert result.final_round.accepted is True
    assert "budget" in result.final_round.accept_reason.lower()
