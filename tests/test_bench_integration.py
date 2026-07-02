"""End-to-end bench integration test.

Uses deep-research-bench's MockEngine + StubJudge with a tiny inline
cases list to exercise run_bench_compare end-to-end."""

from __future__ import annotations

import pytest

# Skip everything in this module if deep-research-bench isn't installed
pytest.importorskip("deep_research_bench")

from deep_research_bench import BenchmarkCase, StubJudge  # noqa: E402
from deep_research_bench.adapters.mock_engine import MockEngine  # noqa: E402

from refinement_loop import RefinementLoop, ThresholdAcceptance  # noqa: E402
from refinement_loop.bench import run_bench_compare  # noqa: E402


def _two_cases():
    return [
        BenchmarkCase(
            id="case-a",
            query="What are the major Python 3.13 features?",
            category="recency",
            must_contain_regex=["3\\.13"],
            min_word_count=20,
        ),
        BenchmarkCase(
            id="case-b",
            query="Compare AVL and Red-Black trees.",
            category="factual-lookup",
        ),
    ]


def test_run_bench_compare_returns_one_row_per_case() -> None:
    loop = RefinementLoop(
        engine=MockEngine(name="mock"),
        judge=StubJudge(
            fixed_score=3
        ),  # always 3 — under threshold=4, forces refinement
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=2,
    )
    report = run_bench_compare(cases=_two_cases(), loop=loop)
    assert len(report.rows) == 2
    assert {r.case_id for r in report.rows} == {"case-a", "case-b"}


def test_report_markdown_renders_summary_and_table() -> None:
    loop = RefinementLoop(
        engine=MockEngine(name="mock"),
        judge=StubJudge(fixed_score=5),  # always 5 — plain accepts immediately
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=3,
    )
    report = run_bench_compare(cases=_two_cases(), loop=loop)
    md = report.to_markdown()
    assert "# Refinement-loop comparison" in md
    assert "Cases: **2**" in md
    assert "| `case-a` |" in md
    assert "| `case-b` |" in md
    # With stub judge always returning 5, both plain and looped should match
    assert "Mean plain score: **5.00**" in md
    assert "Mean looped score: **5.00**" in md


def test_looped_runs_more_rounds_when_score_under_threshold() -> None:
    loop = RefinementLoop(
        engine=MockEngine(name="mock"),
        judge=StubJudge(fixed_score=2),  # always 2 — under any reasonable threshold
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=3,
    )
    report = run_bench_compare(cases=_two_cases(), loop=loop)
    # Each looped run should have spent all 3 rounds since the judge
    # never says "good enough"
    for row in report.rows:
        assert row.looped_rounds == 3
