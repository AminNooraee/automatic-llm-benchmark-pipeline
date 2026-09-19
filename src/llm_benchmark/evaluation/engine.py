"""Anonymous pairwise evaluation through the shared model-client interface."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_benchmark.clients.contracts import ChatMessage, GenerationResult
from llm_benchmark.clients.errors import ModelClientError
from llm_benchmark.clients.interface import ModelClient
from llm_benchmark.config.models import EndpointConfig, EvaluationCriterion
from llm_benchmark.evaluation.artifact_store import JudgeArtifactStore
from llm_benchmark.evaluation.errors import JudgeArtifactError, JudgeOutputError
from llm_benchmark.evaluation.models import (
    AnswerOrder,
    JudgeDecision,
    JudgeEvaluationStatus,
    JudgeInput,
    JudgeResult,
    JudgeResultMetadata,
    JudgeRunResult,
)
from llm_benchmark.evaluation.parser import JudgeOutputParser
from llm_benchmark.evaluation.prompt import JudgePromptTemplate
from llm_benchmark.evaluation.randomization import AnswerRandomizer, AnonymizedAnswers
from llm_benchmark.evaluation.reader import InferenceResponseReader


_LOGGER = logging.getLogger("llm_benchmark.evaluation.engine")


class JudgeEngine:
    def __init__(
        self,
        model_client: ModelClient,
        *,
        response_reader: InferenceResponseReader | None = None,
        output_parser: JudgeOutputParser | None = None,
        randomizer: AnswerRandomizer | None = None,
    ) -> None:
        self._model_client = model_client
        self._response_reader = response_reader or InferenceResponseReader()
        self._output_parser = output_parser or JudgeOutputParser()
        self._randomizer = randomizer

    async def run(
        self,
        *,
        inference_path: Path,
        run_dir: Path,
        judge_model: EndpointConfig,
        prompt_template_path: Path,
        criteria: Sequence[EvaluationCriterion],
        concurrency: int = 1,
        random_seed: int = 42,
        resume: bool = False,
    ) -> JudgeRunResult:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        inputs = self._response_reader.read(inference_path)
        input_by_id = {item.id: item for item in inputs}
        template = JudgePromptTemplate.from_file(prompt_template_path)
        artifact_store = JudgeArtifactStore(run_dir)
        if resume:
            existing = artifact_store.load_results()
        else:
            artifact_store.ensure_empty_for_new_run()
            existing = {}
        self._validate_resume_results(existing, input_by_id)

        randomizer = self._randomizer or AnswerRandomizer(random_seed)
        results: dict[str, JudgeResult] = {}
        work: list[tuple[JudgeInput, AnonymizedAnswers]] = []
        reused = 0
        skipped = 0
        for item in inputs:
            previous = existing.get(item.id)
            if (
                previous is not None
                and previous.metadata.status == JudgeEvaluationStatus.SUCCESS
            ):
                results[item.id] = self._mark_reused(previous)
                reused += 1
                continue
            previous_order = previous.metadata.answer_order if previous else None
            if not item.base_response.strip() or not item.fine_tuned_response.strip():
                order = previous_order or randomizer.arrange(
                    item.base_response, item.fine_tuned_response
                ).order
                results[item.id] = self._skipped_result(
                    item,
                    order=order,
                    judge_model=judge_model.name,
                )
                skipped += 1
                continue
            work.append(
                (item, self._arrange_answers(item, previous_order, randomizer))
            )

        total = len(inputs)
        completed = reused + skipped
        _LOGGER.info(
            "Judge evaluation started",
            extra={
                "completed_samples": completed,
                "event": "evaluation_started",
                "stage": "evaluation",
                "total_samples": total,
            },
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def process(
            item: JudgeInput, arranged: AnonymizedAnswers
        ) -> JudgeResult:
            async with semaphore:
                return await self._evaluate_sample(
                    item,
                    arranged=arranged,
                    template=template,
                    criteria=criteria,
                    judge_model=judge_model,
                )

        def record(result: JudgeResult) -> None:
            nonlocal completed
            results[result.id] = result
            completed += 1
            _LOGGER.info(
                "Judge sample completed",
                extra={
                    "completed_samples": completed,
                    "event": "evaluation_sample_completed",
                    "sample_id": result.id,
                    "sample_status": result.metadata.status.value,
                    "stage": "evaluation",
                    "total_samples": total,
                },
            )

        if concurrency == 1:
            for item, arranged in work:
                record(await process(item, arranged))
        else:
            tasks = [
                asyncio.create_task(process(item, arranged))
                for item, arranged in work
            ]
            try:
                for task in asyncio.as_completed(tasks):
                    record(await task)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        ordered = [results[item.id] for item in inputs]
        artifact_path = artifact_store.write_results(ordered)
        successful = sum(
            item.metadata.status == JudgeEvaluationStatus.SUCCESS for item in ordered
        )
        failed = sum(
            item.metadata.status == JudgeEvaluationStatus.ERROR for item in ordered
        )
        run_result = JudgeRunResult(
            artifact_path=artifact_path,
            total_samples=total,
            successful_samples=successful,
            failed_samples=failed,
            skipped_samples=skipped,
            reused_samples=reused,
            attempted_samples=len(work),
        )
        _LOGGER.info(
            "Judge evaluation completed",
            extra={
                "completed_samples": total,
                "event": "evaluation_completed",
                "stage": "evaluation",
                "total_samples": total,
            },
        )
        return run_result

    async def _evaluate_sample(
        self,
        item: JudgeInput,
        *,
        arranged: AnonymizedAnswers,
        template: JudgePromptTemplate,
        criteria: Sequence[EvaluationCriterion],
        judge_model: EndpointConfig,
    ) -> JudgeResult:
        rendered = template.render(
            question=item.prompt,
            answer_a=arranged.answer_a,
            answer_b=arranged.answer_b,
            criteria=criteria,
        )
        try:
            generated = await self._model_client.generate(
                judge_model,
                [ChatMessage(role="user", content=rendered)],
            )
        except ModelClientError as exc:
            return self._error_result(
                item,
                order=arranged.order,
                judge_model=judge_model.name,
                attempts=max(1, int(getattr(exc, "attempts", 1))),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception as exc:
            return self._error_result(
                item,
                order=arranged.order,
                judge_model=judge_model.name,
                attempts=1,
                error_type=type(exc).__name__,
                error_message="Unexpected judge client error",
            )

        try:
            decision = self._output_parser.parse(generated.content)
        except JudgeOutputError as exc:
            return self._error_result(
                item,
                order=arranged.order,
                judge_model=generated.model_name,
                attempts=generated.attempts,
                error_type=type(exc).__name__,
                error_message=str(exc),
                generated=generated,
            )
        return self._success_result(item, decision, arranged.order, generated)

    @staticmethod
    def _arrange_answers(
        item: JudgeInput,
        previous_order: AnswerOrder | None,
        randomizer: AnswerRandomizer,
    ) -> AnonymizedAnswers:
        if previous_order is None:
            return randomizer.arrange(item.base_response, item.fine_tuned_response)
        answers = {
            "base": item.base_response,
            "fine_tuned": item.fine_tuned_response,
        }
        return AnonymizedAnswers(
            answer_a=answers[previous_order.A.value],
            answer_b=answers[previous_order.B.value],
            order=previous_order,
        )

    @staticmethod
    def _success_result(
        item: JudgeInput,
        decision: JudgeDecision,
        order: AnswerOrder,
        generated: GenerationResult,
    ) -> JudgeResult:
        return JudgeResult(
            id=item.id,
            prompt=item.prompt,
            winner=decision.winner,
            scores=decision.scores,
            criteria_scores=decision.criteria_scores,
            reason=decision.reason,
            metadata=JudgeResultMetadata(
                status=JudgeEvaluationStatus.SUCCESS,
                answer_order=order,
                resolved_winner=order.resolve(decision.winner),
                judge_model=generated.model_name,
                latency_ms=generated.latency_ms,
                attempts=generated.attempts,
                finish_reason=generated.finish_reason,
                usage=generated.usage,
            ),
        )

    @staticmethod
    def _error_result(
        item: JudgeInput,
        *,
        order: AnswerOrder,
        judge_model: str,
        attempts: int,
        error_type: str,
        error_message: str,
        generated: GenerationResult | None = None,
    ) -> JudgeResult:
        return JudgeResult(
            id=item.id,
            prompt=item.prompt,
            metadata=JudgeResultMetadata(
                status=JudgeEvaluationStatus.ERROR,
                answer_order=order,
                judge_model=judge_model,
                latency_ms=generated.latency_ms if generated else None,
                attempts=attempts,
                finish_reason=generated.finish_reason if generated else None,
                usage=generated.usage if generated else None,
                error_type=error_type,
                error_message=error_message,
            ),
        )

    @staticmethod
    def _skipped_result(
        item: JudgeInput,
        *,
        order: AnswerOrder,
        judge_model: str,
    ) -> JudgeResult:
        return JudgeResult(
            id=item.id,
            prompt=item.prompt,
            metadata=JudgeResultMetadata(
                status=JudgeEvaluationStatus.SKIPPED,
                answer_order=order,
                judge_model=judge_model,
                attempts=0,
                error_type="IncompleteInferenceResponse",
                error_message=(
                    "Judge evaluation requires both base and fine-tuned responses"
                ),
            ),
        )

    @staticmethod
    def _mark_reused(result: JudgeResult) -> JudgeResult:
        return result.model_copy(
            update={
                "metadata": result.metadata.model_copy(update={"reused": True})
            }
        )

    @staticmethod
    def _validate_resume_results(
        existing: Mapping[str, JudgeResult],
        inputs: Mapping[str, JudgeInput],
    ) -> None:
        unknown = sorted(set(existing).difference(inputs))
        if unknown:
            raise JudgeArtifactError(
                "Judge artifacts contain sample IDs absent from inference responses: "
                + ", ".join(unknown)
            )
        for sample_id, result in existing.items():
            if result.prompt != inputs[sample_id].prompt:
                raise JudgeArtifactError(
                    f"Judge artifact prompt mismatch for sample '{sample_id}'"
                )
