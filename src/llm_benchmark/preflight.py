"""Small provider-neutral compatibility checks for all logical endpoints."""

from __future__ import annotations

from llm_benchmark.clients.contracts import ChatMessage, RetryPolicy
from llm_benchmark.clients.errors import ModelClientError
from llm_benchmark.clients.interface import ModelClient
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.models import AppConfig, EndpointConfig
from llm_benchmark.evaluation.errors import JudgeOutputError
from llm_benchmark.evaluation.parser import JudgeOutputParser
from llm_benchmark.evaluation.prompt import JudgePromptTemplate
from llm_benchmark.exceptions import BenchmarkError


class PreflightError(BenchmarkError):
    def __init__(self, role: str, reason: str) -> None:
        self.role = role
        super().__init__(f"Preflight failed for {role} endpoint: {reason}")


class EndpointPreflight:
    async def run(
        self, config: AppConfig, *, model_client: ModelClient | None = None
    ) -> None:
        owned = model_client is None
        client = model_client or OpenAICompatibleClient(
            retry_policy=RetryPolicy(
                max_retries=config.runtime.max_retries,
                initial_backoff_seconds=config.runtime.retry_backoff_seconds,
                maximum_backoff_seconds=max(
                    8.0, config.runtime.retry_backoff_seconds
                ),
            )
        )
        try:
            await self._ordinary(client, "base", config.models.base)
            await self._ordinary(client, "fine_tuned", config.models.fine_tuned)
            await self._judge(client, config)
        finally:
            if owned:
                await client.aclose()

    @staticmethod
    async def _ordinary(
        client: ModelClient, role: str, endpoint: EndpointConfig
    ) -> None:
        try:
            await client.generate(
                endpoint,
                [
                    ChatMessage(
                        role="user",
                        content="Reply with the single word: ready",
                    )
                ],
            )
        except ModelClientError as exc:
            raise PreflightError(role, str(exc)) from exc

    @staticmethod
    async def _judge(client: ModelClient, config: AppConfig) -> None:
        try:
            template = JudgePromptTemplate.from_file(config.evaluation.prompt_template)
            prompt = template.render(
                question="Which short answer follows the instruction?",
                answer_a="ready",
                answer_b="not ready",
                criteria=config.evaluation.criteria,
            )
            generated = await client.generate(
                config.models.judge,
                [ChatMessage(role="user", content=prompt)],
            )
            JudgeOutputParser().parse(generated.content)
        except (ModelClientError, JudgeOutputError) as exc:
            raise PreflightError("judge", str(exc)) from exc
