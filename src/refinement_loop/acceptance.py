"""Acceptance protocol — decides when to stop refining.

Built-ins:
    ThresholdAcceptance — stop when judge_score >= threshold
    BudgetAcceptance    — same, plus stop if cumulative latency/cost exceeds budget
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class AcceptanceContext:
    """Info passed to an Acceptance check after each round."""

    round_index: int  # 1-based; round 1 = first attempt
    max_rounds: int
    judge_score: Optional[int]
    cumulative_latency_sec: float


class Acceptance(Protocol):
    """Decide whether the current round's result is "good enough" to
    stop refining. Return True to accept (loop ends), False to refine."""

    name: str

    def accept(self, ctx: AcceptanceContext) -> bool: ...


class ThresholdAcceptance:
    """Accept iff ``judge_score >= threshold``.

    A None judge_score (e.g. no judge supplied) is treated per
    ``accept_when_no_judge``: when True (default), the answer is
    accepted as-is on the first round (no judge ⇒ nothing to improve
    against; refining blindly burns budget). When False, the loop
    proceeds to the full max_rounds.
    """

    name = "threshold"

    def __init__(self, threshold: int = 4, accept_when_no_judge: bool = True) -> None:
        if not (1 <= threshold <= 5):
            raise ValueError("threshold must be in 1..5")
        self.threshold = threshold
        self.accept_when_no_judge = accept_when_no_judge

    def accept(self, ctx: AcceptanceContext) -> bool:
        if ctx.judge_score is None:
            return self.accept_when_no_judge
        return ctx.judge_score >= self.threshold


class BudgetAcceptance:
    """Wraps a threshold check with a wall-clock budget. Once the
    cumulative latency crosses ``max_seconds``, the loop accepts the
    current answer even if the score is below threshold.

    This is the honest "good enough or you're out of time" policy —
    refusing to refine forever when the judge is unsatisfiable saves
    real API budget in production.
    """

    name = "budget"

    def __init__(self, threshold: int = 4, max_seconds: float = 30.0) -> None:
        if not (1 <= threshold <= 5):
            raise ValueError("threshold must be in 1..5")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        self.threshold = threshold
        self.max_seconds = max_seconds

    def accept(self, ctx: AcceptanceContext) -> bool:
        if ctx.cumulative_latency_sec >= self.max_seconds:
            return True
        if ctx.judge_score is None:
            return False
        return ctx.judge_score >= self.threshold
