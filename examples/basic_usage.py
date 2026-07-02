"""Example: run 3 hand-written queries through the refinement loop
with stub defaults (no LLM required). Shows the API in the smallest
possible working form.

Run:
    pip install -e .
    python examples/basic_usage.py
"""

from __future__ import annotations

from deep_research_bench import StubJudge
from deep_research_bench.adapters.mock_engine import MockEngine

from refinement_loop import RefinementLoop, ThresholdAcceptance


def main() -> None:
    loop = RefinementLoop(
        engine=MockEngine(name="mock"),
        # StubJudge(fixed_score=3) always returns 3. Since threshold=4,
        # it will always trigger refinement — great for showing the
        # trace even though we're not measuring real quality gains.
        judge=StubJudge(fixed_score=3),
        acceptance=ThresholdAcceptance(threshold=4),
        max_rounds=3,
    )

    queries = [
        "What is prompt engineering?",
        "How does BM25 differ from TF-IDF?",
        "Explain the vanishing gradient problem.",
    ]

    for q in queries:
        result = loop.run(q)
        print(f"\n=== {q!r} ===")
        print(f"  Total rounds: {result.total_rounds}")
        print(f"  Final score:  {result.final_score}")
        print(f"  Final query:  {result.final_round.query!r}")
        for r in result.rounds:
            marker = "[ok]" if r.accepted else " -> "
            print(
                f"    {marker} round {r.round_index}: "
                f"score={r.judge_score} - {r.accept_reason}"
            )


if __name__ == "__main__":
    main()
