"""Reformulator protocol + StubReformulator.

A reformulator takes the original query + a transcript of prior
(query, answer, judge_score) attempts and produces a new query
that the loop will try next round. Implementations may be:

* pure-text rules (StubReformulator) — no LLM
* LLM-prompted (Ollama, OpenAI, Anthropic, ...) — see ``adapters/``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class PriorAttempt:
    """A single previous round's record, passed to the reformulator so
    it can write a better query than "just append 'be more detailed'"."""

    query: str
    answer_text: str
    judge_score: Optional[int]
    judge_reasoning: str


class Reformulator(Protocol):
    """Anything implementing this can refine queries between rounds."""

    name: str

    def refine(
        self, original_query: str, prior_attempts: List[PriorAttempt]
    ) -> str: ...


class StubReformulator:
    """Deterministic reformulator that appends predefined refinement
    suffixes based on round number. No LLM required. Useful for tests,
    smoke runs, CI, and any environment without API access.

    Suffix progression by round:
      round 1 (first refinement) → "Be specific and cite sources."
      round 2                    → "Include concrete examples and numbers."
      round 3                    → "Address each part of the question separately."
      round >= 4                 → "{previous suffix}"  (saturates)
    """

    name = "stub-reformulator"

    DEFAULT_SUFFIXES = [
        "Be specific and cite sources.",
        "Include concrete examples and numbers.",
        "Address each part of the question separately.",
    ]

    def __init__(self, suffixes: Optional[List[str]] = None) -> None:
        if suffixes is not None and not suffixes:
            raise ValueError("suffixes must be non-empty (or None for defaults)")
        self.suffixes = (
            list(suffixes) if suffixes is not None else list(self.DEFAULT_SUFFIXES)
        )

    def refine(self, original_query: str, prior_attempts: List[PriorAttempt]) -> str:
        # round 1 refinement → suffix index 0 (after 1 attempt)
        # round N refinement → suffix index min(N-1, len-1)
        n = len(prior_attempts)
        if n == 0:
            # Defensive — shouldn't happen since refine is only called
            # AFTER at least one round, but return the original cleanly
            # if a caller misuses the API.
            return original_query
        idx = min(n - 1, len(self.suffixes) - 1)
        suffix = self.suffixes[idx]
        # Use the most recent query as the base; original_query is the
        # very first one. Compound refinement = "previous + new suffix"
        # (avoids losing prior refinements).
        prev_query = prior_attempts[-1].query
        # Strip any trailing duplicate suffix to avoid stacking the same hint.
        if suffix in prev_query:
            return prev_query
        return f"{prev_query} {suffix}".strip()
