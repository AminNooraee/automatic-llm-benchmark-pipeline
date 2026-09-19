"""Concurrent, resumable inference across base and fine-tuned models."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path

from llm_benchmark.clients.contracts import ChatMessage, GenerationResult
from llm_benchmark.clients.errors import ModelClientError
from llm_benchmark.clients.interface import ModelClient
from llm_benchmark.config.models import EndpointConfig
from llm_benchmark.datasets.models import BenchmarkSample
from llm_benchmark.inference.artifact_store import InferenceArtifactStore
from llm_benchmark.inference.dataset_reader import NormalizedDatasetReader
from llm_benchmark.inference.errors import InferenceArtifactError
from llm_benchmark.inference.models import (
    InferenceMetadata,
    InferenceResponse,
    InferenceRunResult,
    ModelResponseMetadata,
    ModelResponseStatus,
    SampleInferenceStatus,
)


_LOGGER = logging.getLogger("llm_benchmark.inference.engine")


class InferenceEngine:
    def __init__(
        self,
        model_client: ModelClient,
        dataset_reader: NormalizedDatasetReader | None = None,
    ) -> None:
        self._model_client = model_client
        self._dataset_reader = dataset_reader or NormalizedDatasetReader()

    async def run(
        self,
        *,
        normalized_dataset_path: Path,
        run_dir: Path,
        base_model: EndpointConfig,
        fine_tuned_model: EndpointConfig,
        concurrency: int = 1,
        resume: bool = True,
    ) -> InferenceRunResult:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        samples = self._dataset_reader.read(normalized_dataset_path)
        sample_by_id = {sample.id: sample for sample in samples}
        artifact_store = InferenceArtifactStore(run_dir)
        if resume:
            existing = artifact_store.load_resume_records()
        else:
            artifact_store.ensure_empty_for_new_run()
            existing = {}
        self._validate_resume_records(existing, sample_by_id)

        responses: dict[str, InferenceResponse] = {}
        pending: list[tuple[BenchmarkSample, InferenceResponse | None]] = []
        skipped = 0
        for sample in samples:
            previous = existing.get(sample.id)
            if previous is not None and previous.metadata.status == SampleInferenceStatus.SUCCESS:
                responses[sample.id] = self._mark_reused(previous)
                skipped += 1
            else:
                pending.append((sample, previous))

        total = len(samples)
        completed = skipped
        _LOGGER.info(
            "Inference started",
            extra={
                "completed_samples": completed,
                "event": "inference_started",
                "stage": "inference",
                "total_samples": total,
            },
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def process(
            sample: BenchmarkSample, previous: InferenceResponse | None
        ) -> InferenceResponse:
            async with semaphore:
                return await self._process_sample(
                    sample,
                    previous,
                    base_model=base_model,
                    fine_tuned_model=fine_tuned_model,
                )

        def record(response: InferenceResponse) -> None:
            nonlocal completed
            responses[response.id] = response
            artifact_store.append_checkpoint(response)
            completed += 1
            _LOGGER.info(
                "Inference sample completed",
                extra={
                    "completed_samples": completed,
                    "event": "inference_sample_completed",
                    "sample_id": response.id,
                    "sample_status": response.metadata.status.value,
                    "stage": "inference",
                    "total_samples": total,
                },
            )

        if concurrency == 1:
            for sample, previous in pending:
                record(await process(sample, previous))
        else:
            tasks = [
                asyncio.create_task(process(sample, previous))
                for sample, previous in pending
            ]
            try:
                for task in asyncio.as_completed(tasks):
                    record(await task)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        ordered = [responses[sample.id] for sample in samples]
        artifact_path = artifact_store.write_responses(ordered)
        successful = sum(
            response.metadata.status == SampleInferenceStatus.SUCCESS
            for response in ordered
        )
        partial = sum(
            response.metadata.status == SampleInferenceStatus.PARTIAL
            for response in ordered
        )
        failed = total - successful - partial
        result = InferenceRunResult(
            artifact_path=artifact_path,
            checkpoint_path=artifact_store.checkpoint_path,
            total_samples=total,
            successful_samples=successful,
            partial_samples=partial,
            failed_samples=failed,
            skipped_samples=skipped,
            attempted_samples=len(pending),
        )
        _LOGGER.info(
            "Inference completed",
            extra={
                "completed_samples": total,
                "event": "inference_completed",
                "stage": "inference",
                "total_samples": total,
            },
        )
        return result

    async def _process_sample(
        self,
        sample: BenchmarkSample,
        previous: InferenceResponse | None,
        *,
        base_model: EndpointConfig,
        fine_tuned_model: EndpointConfig,
    ) -> InferenceResponse:
        message = ChatMessage(role="user", content=sample.prompt)
        base_text, base_metadata = await self._run_model(
            model_role="base",
            model_config=base_model,
            message=message,
            previous_text=previous.base_response if previous else "",
            previous_metadata=previous.metadata.base if previous else None,
            sample_id=sample.id,
        )
        fine_text, fine_metadata = await self._run_model(
            model_role="fine_tuned",
            model_config=fine_tuned_model,
            message=message,
            previous_text=previous.fine_tuned_response if previous else "",
            previous_metadata=previous.metadata.fine_tuned if previous else None,
            sample_id=sample.id,
        )
        successes = sum(
            item.status == ModelResponseStatus.SUCCESS
            for item in (base_metadata, fine_metadata)
        )
        status = (
            SampleInferenceStatus.SUCCESS
            if successes == 2
            else SampleInferenceStatus.PARTIAL
            if successes == 1
            else SampleInferenceStatus.FAILED
        )
        return InferenceResponse(
            id=sample.id,
            prompt=sample.prompt,
            base_response=base_text,
            fine_tuned_response=fine_text,
            metadata=InferenceMetadata(
                status=status,
                base=base_metadata,
                fine_tuned=fine_metadata,
            ),
        )

    async def _run_model(
        self,
        *,
        model_role: str,
        model_config: EndpointConfig,
        message: ChatMessage,
        previous_text: str,
        previous_metadata: ModelResponseMetadata | None,
        sample_id: str,
    ) -> tuple[str, ModelResponseMetadata]:
        if (
            previous_metadata is not None
            and previous_metadata.status == ModelResponseStatus.SUCCESS
        ):
            return previous_text, previous_metadata.model_copy(update={"reused": True})

        _LOGGER.debug(
            "Model inference started",
            extra={
                "event": "model_inference_started",
                "model_role": model_role,
                "sample_id": sample_id,
                "stage": "inference",
            },
        )
        try:
            generated = await self._model_client.generate(model_config, [message])
        except ModelClientError as exc:
            return "", ModelResponseMetadata(
                status=ModelResponseStatus.ERROR,
                model_name=model_config.name,
                attempts=max(1, int(getattr(exc, "attempts", 1))),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception as exc:
            return "", ModelResponseMetadata(
                status=ModelResponseStatus.ERROR,
                model_name=model_config.name,
                attempts=1,
                error_type=type(exc).__name__,
                error_message="Unexpected model client error",
            )
        return generated.content, self._success_metadata(generated)

    @staticmethod
    def _success_metadata(result: GenerationResult) -> ModelResponseMetadata:
        return ModelResponseMetadata(
            status=ModelResponseStatus.SUCCESS,
            model_name=result.model_name,
            latency_ms=result.latency_ms,
            attempts=result.attempts,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )

    @staticmethod
    def _mark_reused(response: InferenceResponse) -> InferenceResponse:
        return response.model_copy(
            update={
                "metadata": response.metadata.model_copy(
                    update={
                        "base": response.metadata.base.model_copy(
                            update={"reused": True}
                        ),
                        "fine_tuned": response.metadata.fine_tuned.model_copy(
                            update={"reused": True}
                        ),
                    }
                )
            }
        )

    @staticmethod
    def _validate_resume_records(
        existing: Mapping[str, InferenceResponse],
        samples: Mapping[str, BenchmarkSample],
    ) -> None:
        unknown = sorted(set(existing).difference(samples))
        if unknown:
            raise InferenceArtifactError(
                "Inference artifacts contain sample IDs absent from the normalized "
                f"dataset: {', '.join(unknown)}"
            )
        for sample_id, response in existing.items():
            if response.prompt != samples[sample_id].prompt:
                raise InferenceArtifactError(
                    f"Inference artifact prompt mismatch for sample '{sample_id}'"
                )
