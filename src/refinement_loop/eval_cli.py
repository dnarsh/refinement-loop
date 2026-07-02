"""CLI: run a deep-research-bench case file through the loop (with stubs
by default; configurable to use Ollama)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refinement-loop-eval",
        description="Run a bench file through the refinement loop and report "
        "the with-loop vs without-loop comparison.",
    )
    p.add_argument(
        "--cases", required=True, help="Path to deep-research-bench cases.json"
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Cap on refinement rounds per case (default 3)",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Judge score >= this stops refinement (default 4)",
    )
    p.add_argument(
        "--engine",
        choices=["mock"],
        default="mock",
        help="Which engine to use (default: mock, which is the deep-research-bench MockEngine)",
    )
    p.add_argument(
        "--reformulator",
        choices=["stub", "ollama"],
        default="stub",
        help="Which reformulator (default: stub — deterministic, no LLM)",
    )
    p.add_argument(
        "--judge",
        choices=["stub", "ollama"],
        default="stub",
        help="Which judge (default: stub — deterministic mid-score)",
    )
    p.add_argument(
        "--ollama-model",
        default="qwen2.5:0.5b",
        help="Ollama model tag (used if --reformulator or --judge is 'ollama')",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write the markdown report here (omit to print to stdout)",
    )
    return p


def _build_engine(name: str):
    if name == "mock":
        from deep_research_bench.adapters.mock_engine import MockEngine

        return MockEngine(name="mock")
    raise ValueError(f"unknown engine: {name}")


def _build_judge(name: str, ollama_model: str):
    if name == "stub":
        from deep_research_bench import StubJudge

        return StubJudge()
    if name == "ollama":
        from refinement_loop.adapters.ollama_adapter import OllamaJudge

        return OllamaJudge(model=ollama_model)
    raise ValueError(f"unknown judge: {name}")


def _build_reformulator(name: str, ollama_model: str):
    if name == "stub":
        from refinement_loop.reformulators import StubReformulator

        return StubReformulator()
    if name == "ollama":
        from refinement_loop.adapters.ollama_adapter import OllamaReformulator

        return OllamaReformulator(model=ollama_model)
    raise ValueError(f"unknown reformulator: {name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from deep_research_bench import load_cases
    except ImportError:
        print(
            "error: deep-research-bench is required.\n"
            "install with: pip install git+https://github.com/aiton08255/deep-research-bench",
            file=sys.stderr,
        )
        return 2

    try:
        cases = load_cases(args.cases)
    except (FileNotFoundError, ValueError) as e:
        print(f"error loading cases: {e}", file=sys.stderr)
        return 2

    from refinement_loop import RefinementLoop, ThresholdAcceptance
    from refinement_loop.bench import run_bench_compare

    loop = RefinementLoop(
        engine=_build_engine(args.engine),
        judge=_build_judge(args.judge, args.ollama_model),
        reformulator=_build_reformulator(args.reformulator, args.ollama_model),
        acceptance=ThresholdAcceptance(threshold=args.threshold),
        max_rounds=args.max_rounds,
    )

    report = run_bench_compare(cases=cases, loop=loop)
    md = report.to_markdown()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
