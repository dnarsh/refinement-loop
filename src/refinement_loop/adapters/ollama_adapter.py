"""Ollama-backed reformulator + judge.

Requires the ``ollama`` Python package and a running ``ollama serve``
with the configured model pulled. Optional — install via:

    pip install refinement-loop[ollama]
    ollama pull qwen2.5:0.5b   # tiny, fast; works for the demo

The implementations are deliberately thin so the prompt-engineering
stays visible: the function bodies ARE the contract with the LLM, so
keep them readable.
"""

from __future__ import annotations

from typing import List, Optional

from refinement_loop.reformulators import PriorAttempt

_REFORMULATOR_PROMPT = """\
You are helping refine a research query that didn't get a good answer.
Given the original query, the last query that was tried, the answer
the engine produced, and the judge's reasoning for the low score,
write a SINGLE better query that addresses the judge's concerns.

Output ONLY the new query — no preamble, no quotation marks, no
explanation. The new query should:
  * stay close in topic to the original
  * be specific about what was missing in the last answer
  * be a single self-contained question

ORIGINAL QUERY: {original}

LAST QUERY TRIED: {last_query}

LAST ANSWER (excerpt): {last_answer_excerpt}

JUDGE'S REASONING: {judge_reasoning}

NEW QUERY:"""


_JUDGE_PROMPT = """\
You are evaluating an answer to a research question. Score the answer
1-5 on a single integer scale:

  1 = Wrong, hallucinated, or off-topic.
  2 = Partially right but missing major requested elements.
  3 = Adequate. Covers the question but lacks depth or specificity.
  4 = Good. Covers the question with appropriate depth.
  5 = Excellent. Comprehensive, accurate, well-cited.

Reply with exactly two lines:

  SCORE: <integer 1-5>
  REASONING: <one sentence>

QUESTION: {query}

ANSWER: {answer}"""


class OllamaReformulator:
    """LLM-prompted query reformulator backed by a local Ollama model.

    Args:
        model: the Ollama model tag (e.g. "qwen2.5:0.5b", "phi3:mini").
            Must already be pulled via ``ollama pull``.
        host: ollama server URL (default uses the env / library default
            of http://127.0.0.1:11434).
        max_answer_excerpt_chars: how much of the prior answer to feed
            back to the reformulator. Long excerpts blow context budget.
    """

    name = "ollama-reformulator"

    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        host: Optional[str] = None,
        max_answer_excerpt_chars: int = 800,
    ) -> None:
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "OllamaReformulator requires the `ollama` package. "
                "Install with: pip install refinement-loop[ollama]"
            ) from e
        self._ollama = ollama
        self.model = model
        self.host = host
        self.max_excerpt = max_answer_excerpt_chars

    def refine(self, original_query: str, prior_attempts: List[PriorAttempt]) -> str:
        if not prior_attempts:
            return original_query
        last = prior_attempts[-1]
        excerpt = (last.answer_text or "")[: self.max_excerpt]
        prompt = _REFORMULATOR_PROMPT.format(
            original=original_query,
            last_query=last.query,
            last_answer_excerpt=excerpt,
            judge_reasoning=last.judge_reasoning or "(no judge reasoning)",
        )
        client_kwargs = {"host": self.host} if self.host else {}
        client = self._ollama.Client(**client_kwargs) if client_kwargs else self._ollama
        resp = client.generate(model=self.model, prompt=prompt, stream=False)
        text = (
            resp.get("response") if isinstance(resp, dict) else resp.response
        ).strip()
        # Reject empties / runaway outputs — fall back to "last query +
        # judge reasoning as suffix" so the loop keeps moving.
        if not text:
            return f"{last.query} ({last.judge_reasoning})".strip()
        # If the model wrapped the answer in quotes, strip them.
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1]
        return text


class OllamaJudge:
    """LLM-as-judge backed by a local Ollama model. Returns objects
    compatible with deep_research_bench's JudgeResult shape (the
    refinement loop uses duck-typing for that)."""

    name = "ollama-judge"

    def __init__(self, model: str = "qwen2.5:0.5b", host: Optional[str] = None) -> None:
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "OllamaJudge requires the `ollama` package. "
                "Install with: pip install refinement-loop[ollama]"
            ) from e
        self._ollama = ollama
        self.model = model
        self.host = host

    def judge(self, case, answer) -> "_OllamaJudgeResult":
        prompt = _JUDGE_PROMPT.format(
            query=getattr(case, "query", ""),
            answer=getattr(answer, "text", ""),
        )
        client_kwargs = {"host": self.host} if self.host else {}
        client = self._ollama.Client(**client_kwargs) if client_kwargs else self._ollama
        resp = client.generate(model=self.model, prompt=prompt, stream=False)
        text = resp.get("response") if isinstance(resp, dict) else resp.response
        score, reasoning = _parse_judge_response(text or "")
        return _OllamaJudgeResult(
            score=score, reasoning=reasoning, judge_name=self.name
        )


def _parse_judge_response(text: str) -> tuple:
    """Lenient parser for ``SCORE: N\\nREASONING: ...``.

    Clamps out-of-range scores to [1, 5] (rather than raising, which
    would abort the whole loop on one bad LLM response). Returns
    ``(3, "<parse-failure>")`` if no SCORE line found — gives the loop
    a neutral default rather than crashing.
    """
    import re

    score_m = re.search(r"SCORE\s*[:=]\s*(\d+)", text, re.IGNORECASE)
    if not score_m:
        return 3, "(parse failure: SCORE line not found)"
    raw_score = int(score_m.group(1))
    score = max(1, min(5, raw_score))
    reason_m = re.search(
        r"REASONING\s*[:=]\s*(.+?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL
    )
    reasoning = reason_m.group(1).strip() if reason_m else ""
    return score, reasoning


class _OllamaJudgeResult:
    """Tiny duck-type matching deep_research_bench.JudgeResult."""

    def __init__(self, score: int, reasoning: str, judge_name: str) -> None:
        self.score = score
        self.reasoning = reasoning
        self.judge_name = judge_name
