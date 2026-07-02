"""Adapters for real LLM backends.

Currently shipped:
    OllamaReformulator    — Ollama-backed reformulator (any local model)
    OllamaJudge           — Ollama-backed LLM-as-judge

Why Ollama and not Anthropic/OpenAI?
    Ollama runs entirely locally — no API keys, no quotas, no spend.
    It's the easiest "real LLM" any developer can stand up.

    The same pattern works for Anthropic/OpenAI — see ``OllamaJudge``
    for the prompt-builder + response-parser shape and write your own
    adapter against your provider's SDK.
"""

from refinement_loop.adapters.ollama_adapter import OllamaJudge, OllamaReformulator

__all__ = ["OllamaJudge", "OllamaReformulator"]
