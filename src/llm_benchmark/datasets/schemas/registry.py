"""Registry for built-in schema adapters."""

from __future__ import annotations

from collections.abc import Iterable

from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.chatml_adapter import ChatMLSchemaAdapter
from llm_benchmark.datasets.schemas.instruction_adapter import (
    InstructionSchemaAdapter,
)
from llm_benchmark.datasets.schemas.openai_messages_adapter import (
    OpenAIMessagesSchemaAdapter,
)
from llm_benchmark.datasets.schemas.prompt_adapter import PromptSchemaAdapter
from llm_benchmark.datasets.schemas.question_answer_adapter import (
    QuestionAnswerSchemaAdapter,
)
from llm_benchmark.datasets.schemas.sharegpt_adapter import ShareGPTSchemaAdapter


class SchemaAdapterRegistry:
    def __init__(self, adapters: Iterable[SchemaAdapter] | None = None) -> None:
        self._adapters = tuple(
            adapters
            or (
                PromptSchemaAdapter(),
                QuestionAnswerSchemaAdapter(),
                InstructionSchemaAdapter(),
                OpenAIMessagesSchemaAdapter(),
                ChatMLSchemaAdapter(),
                ShareGPTSchemaAdapter(),
            )
        )

    @property
    def adapters(self) -> tuple[SchemaAdapter, ...]:
        return self._adapters

    @property
    def possible_formats(self) -> tuple[str, ...]:
        return tuple(adapter.schema.value for adapter in self._adapters)

