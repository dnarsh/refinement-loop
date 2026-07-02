"""The core RefinementLoop orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from refinement_loop.acceptance import (
    Acceptance,
    AcceptanceContext,
    ThresholdAcceptance,
)
from refinement_loop.reformulators import PriorAttempt, Reformulator, StubReformulator


# ── Engine + Judge protocols (duck-typed from deep_research_bench) ──
#
# We DON'T import EngineAnswer / JudgeResult from deep_research_bench
# here because the core loop should be usable without the bench dep.
# Instead we duck-type: anything with the right attribute shape works.


class _EngineLike(Protocol):
    """Protocol the loop expects from an engine. Compatible with
    deep_research_bench.Engine."""

    name: str

    def run(
        self, query: str
    ) -> Any:  # returns something with .text, .citations, .metadata
        ...


class _JudgeLike(Protocol):
    """Protocol the loop expects from a judge. Compatible with
    deep_research_bench.Judge. May be None (no scoring)."""

    name: str

    def judge(
        self, case: Any, answer: Any
    ) -> Any:  # returns something with .score, .reasoning
        ...


@dataclass
class RoundRecord:
    """Per-round trace for inspection."""

    round_index: int  # 1-based
    query: str
    answer_text: str
    answer_citations: List[str] = field(default_factory=list)
    answer_metadata: Dict[str, Any] = field(default_factory=dict)
    judge_score: Optional[int] = None
    judge_reasoning: str = ""
    latency_sec: float = 0.0
    accepted: bool = False
    accept_reason: str = ""


@dataclass
class LoopResult:
    """Final result of a RefinementLoop.run() call."""

    original_query: str
    rounds: List[RoundRecord] = field(default_factory=list)
    final_round: Optional[RoundRecord] = None
    total_latency_sec: float = 0.0

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def final_answer_text(self) -> str:
        return self.final_round.answer_text if self.final_round else ""

    @property
    def final_score(self) -> Optional[int]:
        return self.final_round.judge_score if self.final_round else None


class RefinementLoop:
    """Iterative refine-until-accepted loop.

    Args:
        engine: anything with ``run(query) -> EngineAnswer-like``
            (text/citations/metadata).
        judge: optional — anything with ``judge(case, answer) -> JudgeResult-like``
            (score/reasoning). When None, the loop runs exactly once
            with no scoring (acceptance is then a no-op).
        reformulator: anything with ``refine(query, prior_attempts) -> str``.
            Defaults to [`StubReformulator`].
        acceptance: anything with ``accept(ctx) -> bool``. Defaults to
            [`ThresholdAcceptance`] with threshold=4.
        max_rounds: hard cap on total attempts. Default 3.

    Notes:
        * Engine exceptions propagate. The loop is intentionally NOT a
          retry-on-failure mechanism — it's a refine-on-low-score
          mechanism. If your engine is flaky, wrap it yourself.
        * The "case" passed to the judge is a duck-typed stand-in:
          an object with ``.query`` and ``.judge_criteria`` set so the
          judge has enough context. The loop builds it on the fly.
    """

    def __init__(
        self,
        engine: _EngineLike,
        judge: Optional[_JudgeLike] = None,
        reformulator: Optional[Reformulator] = None,
        acceptance: Optional[Acceptance] = None,
        max_rounds: int = 3,
        judge_criteria: str = "",
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.engine = engine
        self.judge = judge
        self.reformulator = reformulator or StubReformulator()
        self.acceptance = acceptance or ThresholdAcceptance(threshold=4)
        self.max_rounds = max_rounds
        self.judge_criteria = judge_criteria

    def run(self, original_query: str) -> LoopResult:
        if not original_query or not original_query.strip():
            raise ValueError("original_query must be non-empty")

        result = LoopResult(original_query=original_query)
        prior_attempts: List[PriorAttempt] = []
        current_query = original_query
        cumulative_latency = 0.0

        for round_index in range(1, self.max_rounds + 1):
            started = time.monotonic()
            answer = self.engine.run(current_query)
            elapsed = time.monotonic() - started
            cumulative_latency += elapsed

            record = RoundRecord(
                round_index=round_index,
                query=current_query,
                answer_text=getattr(answer, "text", ""),
                answer_citations=list(getattr(answer, "citations", []) or []),
                answer_metadata=dict(getattr(answer, "metadata", {}) or {}),
                latency_sec=elapsed,
            )

            if self.judge is not None:
                case_stub = _CaseStub(
                    query=current_query, judge_criteria=self.judge_criteria
                )
                jres = self.judge.judge(case_stub, answer)
                record.judge_score = getattr(jres, "score", None)
                record.judge_reasoning = getattr(jres, "reasoning", "") or ""

            ctx = AcceptanceContext(
                round_index=round_index,
                max_rounds=self.max_rounds,
                judge_score=record.judge_score,
                cumulative_latency_sec=cumulative_latency,
            )
            accepted = self.acceptance.accept(ctx)
            record.accepted = accepted
            record.accept_reason = self._accept_reason(ctx, accepted, round_index)

            result.rounds.append(record)

            if accepted or round_index == self.max_rounds:
                result.final_round = record
                break

            # Refine for the next round
            prior_attempts.append(
                PriorAttempt(
                    query=record.query,
                    answer_text=record.answer_text,
                    judge_score=record.judge_score,
                    judge_reasoning=record.judge_reasoning,
                )
            )
            current_query = self.reformulator.refine(original_query, prior_attempts)

        result.total_latency_sec = cumulative_latency
        return result

    def _accept_reason(
        self, ctx: AcceptanceContext, accepted: bool, round_index: int
    ) -> str:
        if accepted:
            if ctx.judge_score is None:
                return "no judge; accepted as-is"
            return f"score {ctx.judge_score} meets acceptance ({self.acceptance.name})"
        if round_index >= self.max_rounds:
            return f"max_rounds={self.max_rounds} reached; final"
        score_part = (
            f"score {ctx.judge_score}" if ctx.judge_score is not None else "no score"
        )
        return f"{score_part} below acceptance; refining"


@dataclass(frozen=True)
class _CaseStub:
    """Minimal case-like object passed to judges. Mirrors the fields a
    judge built for deep_research_bench actually reads."""

    query: str
    judge_criteria: str = ""
    id: str = "_inline"
    category: str = ""
