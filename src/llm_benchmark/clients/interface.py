"""Unified model-client interface consumed by later pipeline phases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from llm_benchmark.clients.contracts import ChatMessage, GenerationResult
from llm_benchmark.config.models import EndpointConfig


class ModelClient(Protocol):
    async def generate(
        self,
        model_config: EndpointConfig,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
    ) -> GenerationResult:
        """Generate one non-streaming chat completion."""

    async def aclose(self) -> None:
        """Release network resources."""

