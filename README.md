# refinement-loop

A small (~600 LOC, one runtime dep, Python 3.8+) reference implementation of the **answer-then-refine loop** — the pattern where a research engine answers a query, a judge scores the answer, and if the score is too low, a reformulator rewrites the query for the next attempt.

Composes cleanly with [`deep-research-bench`](https://github.com/aiton08255/deep-research-bench) for measurable A/B comparison. Ships with **stub defaults that run today with zero API keys** and an optional Ollama adapter for real local-LLM refinement.

Extracted from a private long-running research system where iterative refinement measurably beat single-shot answering on questions with a specific structure (multi-part synthesis, citation-density-sensitive queries, "distinguish demo from proven" adversarial cases). The pattern is generalized here; the private engine and eval data stay closed.

> **Status (v0.1.0):** Source-only release. Not yet on PyPI. Install from GitHub (see [Install](#install)). Publish planned.

---

## What this is

A `RefinementLoop` class that wraps three pluggable components:

1. **Engine** — anything with `run(query) -> EngineAnswer-like`. The deep-research-bench engine protocol works directly; anything with `.text`, `.citations`, `.metadata` attributes works via duck-typing.
2. **Judge** — anything with `judge(case, answer) -> JudgeResult-like` (has `.score`, `.reasoning`). Optional. Without a judge, the loop runs exactly once (single-shot fallback).
3. **Reformulator** — anything with `refine(original_query, prior_attempts) -> str`. Given the history of what was tried and how the judge scored it, produce a better query.

Plus an **Acceptance** policy — decides when to stop refining (built-ins: threshold-only, threshold-with-time-budget).

## What this is NOT — read this first

| If you need... | Use... |
|---|---|
| **A retry-on-failure mechanism** (engine crashes → retry) | Wrap the engine in your own try/except. This loop refines on LOW SCORES, not on exceptions. |
| **A production agent framework** with tool use, memory, multi-step planning | [`LangChain`](https://python.langchain.com), [`LangGraph`](https://langchain-ai.github.io/langgraph/), [`Autogen`](https://microsoft.github.io/autogen/), [`DSPy`](https://dspy.ai/) |
| **Automatic prompt optimization** ("compile" prompts against a dataset) | [`DSPy`](https://dspy.ai/) is the canonical option and does this MUCH better than an ad-hoc loop. |
| **Streaming / async LLM calls** | This loop is sync. Wrap it in `asyncio.to_thread` if you need to. |
| **A pre-built leaderboard** showing "loop vs no-loop" numbers across public benchmarks | Not shipped. Frontier engines' quotas make honest public numbers expensive to produce. Wire in your own engine + judge and run `refinement-loop-eval` yourself. |

**Use this when**: you already have an engine + judge, and you want to measure whether adding a refine step improves the quality on your specific case set.

---

## Install

Source-only until v0.1.0 lands on PyPI. Install BOTH packages (the bench is a runtime dep):

```bash
pip install git+https://github.com/aiton08255/deep-research-bench
pip install git+https://github.com/aiton08255/refinement-loop
```

Or clone and install editable for local hacking:

```bash
git clone https://github.com/aiton08255/deep-research-bench
git clone https://github.com/aiton08255/refinement-loop
pip install -e ./deep-research-bench
pip install -e "./refinement-loop[dev]"
```

For the Ollama adapter (recommended for real refinement):

```bash
pip install "refinement-loop[ollama]"
ollama pull qwen2.5:0.5b   # tiny + fast; works well for the reformulator role
```

---

## Quick start — see the loop in action in 30 seconds

The package ships a CLI that runs the [10 public benchmark cases from deep-research-bench](https://github.com/aiton08255/deep-research-bench/blob/main/cases/public-10.json) through the loop with `MockEngine` + `StubJudge`. The mock engine intentionally returns gibberish — the point is the report SHAPE, not the score.

```bash
refinement-loop-eval --cases cases/public-10.json --max-rounds 3 --threshold 4
```

Real-engine + real-judge run using Ollama:

```bash
ollama serve   # if not already running
ollama pull qwen2.5:0.5b
refinement-loop-eval \
    --cases cases/public-10.json \
    --reformulator ollama --judge ollama \
    --ollama-model qwen2.5:0.5b \
    --max-rounds 3 --threshold 4 \
    --output results/
```

Reads out as a markdown table showing plain-vs-looped scores per case + means + latency delta.

## Python API

```python
from deep_research_bench import StubJudge, load_cases
from deep_research_bench.adapters.mock_engine import MockEngine

from refinement_loop import RefinementLoop, ThresholdAcceptance
from refinement_loop.bench import run_bench_compare

loop = RefinementLoop(
    engine=MockEngine(name="mock"),      # replace with your real engine
    judge=StubJudge(fixed_score=3),      # replace with a real judge for real numbers
    acceptance=ThresholdAcceptance(threshold=4),
    max_rounds=3,
)

cases = load_cases("cases/public-10.json")
report = run_bench_compare(cases=cases, loop=loop)
print(report.to_markdown())
print(f"Mean delta: {report.mean_delta:+.2f} judge-points")
```

## Wiring a real engine

Any object with a `name` and `run(query)` method works. See the [deep-research-bench README](https://github.com/aiton08255/deep-research-bench#wiring-a-real-engine) for the full engine-adapter pattern (~30 lines for Perplexity / OpenAI / Anthropic / your own).

## Wiring a real judge

For the Ollama-backed judge (no API keys, runs locally):

```python
from refinement_loop.adapters import OllamaJudge, OllamaReformulator

loop = RefinementLoop(
    engine=MockEngine(name="mock"),
    judge=OllamaJudge(model="qwen2.5:0.5b"),
    reformulator=OllamaReformulator(model="qwen2.5:0.5b"),
    acceptance=ThresholdAcceptance(threshold=4),
    max_rounds=3,
)
```

For hosted-API judges (Anthropic, OpenAI), see the Ollama adapter's `_JUDGE_PROMPT` and `_parse_judge_response` — copy the prompt + parser and swap in your provider's SDK call. It's ~20 lines.

---

## The loop in one paragraph

```
For round in 1..max_rounds:
    answer = engine.run(current_query)
    score  = judge.judge(case, answer)   # optional
    if acceptance.accept(score, cumulative_latency): stop
    current_query = reformulator.refine(original_query, prior_attempts)
```

That's it. The whole loop is ~120 LOC (`src/refinement_loop/loop.py`). The rest is protocols, built-in acceptances, the stub reformulator, and the bench-integration helpers.

---

## Reformulators

| Reformulator | Deps | When to use |
|---|---|---|
| `StubReformulator` (default) | none | Tests, smoke runs, CI. Rotates through 3 fixed suffixes: "Be specific and cite sources.", "Include concrete examples and numbers.", "Address each part of the question separately." Deterministic. |
| `OllamaReformulator` | `ollama` (local) | Real refinement, no API keys. Uses a small local model (qwen2.5:0.5b works). |
| **Your own** | your own | Implement `refine(original_query, prior_attempts) -> str`. Two lines of interface. |

---

## Acceptances

| Acceptance | Behavior |
|---|---|
| `ThresholdAcceptance(threshold=N)` | Accept iff `judge_score >= N`. Default `N=4`. When there's no judge, accepts on round 1 (configurable). |
| `BudgetAcceptance(threshold=N, max_seconds=T)` | Same as threshold, PLUS accepts unconditionally once cumulative latency exceeds `T`. Prevents runaway spend on unsatisfiable queries. |

---

## Limitations — the honest list

1. **The loop can only improve what the judge can measure.** If your judge is noisy or systematically miscalibrated, the loop refines toward the judge's preferences, not toward better answers. This is the classic Goodhart problem. Fix by improving the judge, not by adding loop rounds.

2. **No batched inference.** Each round is one engine call. If your engine supports batching, you could run N variant queries in parallel per round and pick the best — this loop doesn't do that. See DSPy for that pattern.

3. **The stub reformulator doesn't actually reformulate — it appends suffixes.** It's for testing the loop's control flow, not for real quality gains. Real refinement needs an LLM reformulator.

4. **Cost blows up linearly with `max_rounds`.** A 3-round loop costs 3x an engine call + 3x a judge call (if the loop doesn't accept early). Budget accordingly.

5. **The Ollama judge parser is lenient.** If the model returns garbage that doesn't contain "SCORE: N", the parser returns a neutral 3 rather than raising. This is intentional (one bad LLM response shouldn't abort a 10-case benchmark) but it means silent score inflation is possible. Spot-check judge outputs in production.

6. **No engine call cancellation.** If your engine takes 30s and the budget acceptance is 10s, the loop still waits for the 30s answer before checking the budget. Budget is per-cycle-boundary, not mid-cycle.

7. **The reformulator can produce a worse query.** No safeguard for "if round 2 scored lower than round 1, revert". The loop always uses the latest attempt. Fix by adding a "best-so-far" tracker in your own subclass or accepting the tradeoff.

8. **The `_CaseStub` passed to inline judges is minimal.** If your custom judge reads case attributes beyond `.query` and `.judge_criteria` (e.g. `min_word_count`, `must_contain_regex`), pass a real case object instead of relying on the loop's stub.

9. **`OllamaReformulator` and `OllamaJudge` require a running `ollama serve`.** The library will raise on the first call if Ollama isn't reachable. No retry logic.

10. **25 tests cover: loop control flow, all four built-in acceptances, StubReformulator, and bench-integration. They do NOT cover: real Ollama calls, real Anthropic/OpenAI adapters, multi-threaded / async use, or extreme edge cases (queries with millions of tokens).** Bug reports welcome.

---

## Alternatives — when NOT to use this

| Tool | Use it instead when |
|---|---|
| [`DSPy`](https://dspy.ai) | You want automatic prompt/example optimization over a training set. DSPy's `MIPRO` and `BootstrapFewShot` do more sophisticated versions of this pattern. |
| [`LangChain`](https://python.langchain.com) `RunnableRetry` / retry chains | You're already in the LangChain ecosystem and want a retry-on-condition abstraction. |
| [`Guidance`](https://github.com/guidance-ai/guidance) / constrained decoding | You want to enforce structure at generation time rather than refining afterward. |
| Just a plain `while` loop | Your refinement logic is 5 lines and pulling in a dep isn't worth it. Fair. |

**Why this exists when DSPy does more**: DSPy is a training-set-driven prompt optimization framework — it needs example data and takes minutes to compile. This crate is for the online case: one query at a time, use it whenever you have an engine + judge, no compilation step. Different tradeoffs. If your use case fits DSPy, use DSPy.

---

## Run the tests

```bash
git clone https://github.com/aiton08255/refinement-loop
cd refinement-loop
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Current status: **25 passed in ~0.4s** on Python 3.12 / Windows 11.

The key tests:
- `test_loop_refines_then_accepts_on_better_round` — round 1 fails, round 2 accepts. Verifies the reformulator is called between rounds.
- `test_loop_respects_max_rounds_when_nothing_meets_threshold` — never accepts, caps at max_rounds.
- `test_loop_with_budget_acceptance_stops_on_budget` — slow engine + tight budget → budget trips before threshold.
- `test_looped_runs_more_rounds_when_score_under_threshold` — end-to-end via `run_bench_compare` with `StubJudge(fixed_score=2)` forcing all 3 rounds.

---

## Provenance

Pattern extracted from a private long-running research system that used iterative refinement to close a quality gap against several frontier engines on questions with a specific structure. The private eval showed a mean +0.4 judge-point improvement (on a 1-5 scale) at a 3x cost multiplier — worth it for high-value queries, not worth it for cheap lookups.

The refinement discipline (cheap first-pass → judge-gated refine → hard budget cap) is the pattern that made this work in practice; the crate publishes exactly that scaffold. Your numbers will depend on your engine, your judge, and the shape of your queries.

The private engine, private eval data, and private cost model stay closed. This crate ships the framework — you plug in what you have.

---

## License

MIT. See `LICENSE`.

## Contributing

PRs welcome for:

- Real bugs (with a reproducer)
- Additional built-in reformulators / acceptances with a clear, narrow purpose
- Cross-platform test fixes

Please don't submit PRs that:
- Bundle a specific provider's SDK as a required dep (adapters go in `adapters/` and stay optional)
- Add async / streaming as the default (acceptable as opt-in addon)
- Rename the public API without a very good reason

Open an [issue](https://github.com/aiton08255/refinement-loop/issues) before a large PR.
