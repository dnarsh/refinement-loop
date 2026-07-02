"""deep-research-bench integration helpers.

Lets you measure the value of the refinement loop on a benchmark suite
by comparing engine-alone vs engine+loop on the same case set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from refinement_loop.loop import LoopResult, RefinementLoop


@dataclass
class ComparisonRow:
    case_id: str
    category: str
    plain_score: Optional[int]
    looped_score: Optional[int]
    plain_text: str
    looped_text: str
    looped_rounds: int
    plain_latency_sec: float
    looped_latency_sec: float

    @property
    def delta(self) -> Optional[int]:
        if self.plain_score is None or self.looped_score is None:
            return None
        return self.looped_score - self.plain_score


@dataclass
class ComparisonReport:
    rows: List[ComparisonRow] = field(default_factory=list)

    @property
    def mean_plain(self) -> Optional[float]:
        scored = [r.plain_score for r in self.rows if r.plain_score is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def mean_looped(self) -> Optional[float]:
        scored = [r.looped_score for r in self.rows if r.looped_score is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def mean_delta(self) -> Optional[float]:
        deltas = [r.delta for r in self.rows if r.delta is not None]
        return sum(deltas) / len(deltas) if deltas else None

    @property
    def total_plain_latency(self) -> float:
        return sum(r.plain_latency_sec for r in self.rows)

    @property
    def total_looped_latency(self) -> float:
        return sum(r.looped_latency_sec for r in self.rows)

    def to_markdown(self) -> str:
        lines = []
        lines.append("# Refinement-loop comparison")
        lines.append("")
        lines.append(f"- Cases: **{len(self.rows)}**")
        if self.mean_plain is not None:
            lines.append(f"- Mean plain score: **{self.mean_plain:.2f}**")
        if self.mean_looped is not None:
            lines.append(f"- Mean looped score: **{self.mean_looped:.2f}**")
        if self.mean_delta is not None:
            sign = "+" if self.mean_delta >= 0 else ""
            lines.append(
                f"- Mean delta (looped - plain): **{sign}{self.mean_delta:.2f}**"
            )
        lines.append(
            f"- Total latency - plain: {self.total_plain_latency:.1f}s, "
            f"looped: {self.total_looped_latency:.1f}s"
        )
        lines.append("")
        lines.append(
            "| Case | Category | Plain | Looped | Delta | Rounds | Plain (s) | Looped (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in self.rows:
            plain = str(r.plain_score) if r.plain_score is not None else "-"
            looped = str(r.looped_score) if r.looped_score is not None else "-"
            delta = (
                f"{'+' if (r.delta or 0) >= 0 else ''}{r.delta}"
                if r.delta is not None
                else "-"
            )
            lines.append(
                f"| `{r.case_id}` | {r.category} | {plain} | {looped} | {delta} | "
                f"{r.looped_rounds} | {r.plain_latency_sec:.2f} | {r.looped_latency_sec:.2f} |"
            )
        return "\n".join(lines) + "\n"


def run_bench_compare(
    cases: Sequence,  # list of deep_research_bench.BenchmarkCase
    loop: RefinementLoop,
    plain_loop_factory=None,
) -> ComparisonReport:
    """Run each case twice: once through the engine alone (a no-judge,
    1-round loop) and once through the supplied refinement loop. Return
    a side-by-side report.

    The plain run uses the same engine + judge as ``loop`` so the score
    comparison is apples-to-apples. The reformulator is irrelevant for
    the plain run because it never refines.

    Args:
        cases: iterable of BenchmarkCase (or any duck-type with `.id`,
            `.query`, `.category`).
        loop: the configured RefinementLoop to evaluate.
        plain_loop_factory: optional callable producing a "plain" loop
            (defaults to a copy of ``loop`` with max_rounds=1 and the
            same judge/engine).
    """
    if plain_loop_factory is None:

        def plain_loop_factory():
            return RefinementLoop(
                engine=loop.engine,
                judge=loop.judge,
                reformulator=loop.reformulator,
                acceptance=loop.acceptance,
                max_rounds=1,
                judge_criteria=loop.judge_criteria,
            )

    rows: List[ComparisonRow] = []
    for case in cases:
        criteria = getattr(case, "judge_criteria", "")
        loop.judge_criteria = criteria  # pass per-case criteria to judge
        plain = plain_loop_factory()
        plain.judge_criteria = criteria

        plain_result: LoopResult = plain.run(case.query)
        looped_result: LoopResult = loop.run(case.query)

        rows.append(
            ComparisonRow(
                case_id=getattr(case, "id", "unknown"),
                category=getattr(case, "category", ""),
                plain_score=plain_result.final_score,
                looped_score=looped_result.final_score,
                plain_text=plain_result.final_answer_text,
                looped_text=looped_result.final_answer_text,
                looped_rounds=looped_result.total_rounds,
                plain_latency_sec=plain_result.total_latency_sec,
                looped_latency_sec=looped_result.total_latency_sec,
            )
        )

    return ComparisonReport(rows=rows)
