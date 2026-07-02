"""refinement-loop — answer-then-refine pattern, composable with deep-research-bench.

Public API:
    RefinementLoop          — the orchestrator
    LoopResult, RoundRecord — typed result + per-round trace
    Reformulator            — protocol for query-rewriting
    Acceptance              — protocol for "good enough" decisions
    StubReformulator        — deterministic stub (no LLM required)
    ThresholdAcceptance     — accept iff judge_score >= threshold
    BudgetAcceptance        — same, but caps cost/latency budget too
    run_bench_compare       — bench-driven helper: with-loop vs without-loop
"""

from refinement_loop.acceptance import (
    Acceptance,
    BudgetAcceptance,
    ThresholdAcceptance,
)
from refinement_loop.loop import LoopResult, RefinementLoop, RoundRecord
from refinement_loop.reformulators import Reformulator, StubReformulator

__all__ = [
    "Acceptance",
    "BudgetAcceptance",
    "LoopResult",
    "RefinementLoop",
    "Reformulator",
    "RoundRecord",
    "StubReformulator",
    "ThresholdAcceptance",
]
__version__ = "0.1.0"


def __getattr__(name):
    """Lazy load helpers that require `deep_research_bench` so the core
    loop can be imported without the bench dep installed."""
    if name == "run_bench_compare":
        from refinement_loop.bench import run_bench_compare

        return run_bench_compare
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
