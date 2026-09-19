"""Provider-independent OpenAI-compatible model communication."""

from llm_benchmark.clients.contracts import (
    ChatMessage,
    GenerationResult,
    GenerationUsage,
    RetryPolicy,
)
from llm_benchmark.clients.interface import ModelClient
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient

__all__ = [
    "ChatMessage",
    "GenerationResult",
    "GenerationUsage",
    "ModelClient",
    "OpenAICompatibleClient",
    "RetryPolicy",
]

