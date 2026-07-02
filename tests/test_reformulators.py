"""StubReformulator tests."""

from __future__ import annotations

import pytest

from refinement_loop.reformulators import PriorAttempt, StubReformulator


def _attempt(query: str, score=None) -> PriorAttempt:
    return PriorAttempt(
        query=query, answer_text="", judge_score=score, judge_reasoning=""
    )


def test_stub_reformulator_empty_prior_returns_original() -> None:
    r = StubReformulator()
    assert r.refine("hello?", []) == "hello?"


def test_stub_reformulator_appends_first_suffix_on_round_1() -> None:
    r = StubReformulator()
    result = r.refine("What is X?", [_attempt("What is X?")])
    assert result.startswith("What is X?")
    assert "Be specific and cite sources." in result


def test_stub_reformulator_compounds_suffix_using_last_query() -> None:
    """When we have 2 prior attempts, the reformulator should build on
    the most recent query (which already contains round-1 suffix)."""
    r = StubReformulator()
    after_round_1 = "What is X? Be specific and cite sources."
    result = r.refine(
        "What is X?",
        [_attempt("What is X?"), _attempt(after_round_1)],
    )
    assert "Be specific and cite sources." in result
    assert "Include concrete examples and numbers." in result


def test_stub_reformulator_does_not_stack_duplicate_suffix() -> None:
    """If the suffix is already present in the last query, don't append
    it again (avoids 'foo foo foo' style buildups)."""
    r = StubReformulator()
    # 4th refinement saturates to the last (3rd-index) suffix because
    # only 3 default suffixes exist.
    attempts = [
        _attempt("q"),
        _attempt("q a"),
        _attempt("q a b"),
        _attempt("q a b Address each part of the question separately."),
    ]
    result = r.refine("q", attempts)
    assert result.count("Address each part of the question separately.") == 1


def test_stub_reformulator_rejects_empty_suffix_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StubReformulator(suffixes=[])


def test_stub_reformulator_custom_suffixes() -> None:
    r = StubReformulator(suffixes=["FOO", "BAR"])
    result = r.refine("q", [_attempt("q")])
    assert "FOO" in result
