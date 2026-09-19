"""Top-level application orchestration for implemented phases."""

from __future__ import annotations

from pathlib import Path

from llm_benchmark.clients.contracts import RetryPolicy
from llm_benchmark.clients.interface import ModelClient
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.models import AppConfig
from llm_benchmark.datasets.models import DatasetIngestionResult
from llm_benchmark.datasets.service import DatasetIngestionService
from llm_benchmark.domain.runs import RunContext
from llm_benchmark.evaluation.engine import JudgeEngine
from llm_benchmark.evaluation.artifact_store import JudgeArtifactStore
from llm_benchmark.evaluation.models import JudgeRunResult
from llm_benchmark.evaluation.reader import InferenceResponseReader
from llm_benchmark.inference.engine import InferenceEngine
from llm_benchmark.inference.models import InferenceRunResult
from llm_benchmark.metrics.calculator import BenchmarkMetricsCalculator
from llm_benchmark.reporting.generator import ReportGenerator
from llm_benchmark.reporting.models import ReportGenerationResult
from llm_benchmark.runs.manager import RunManager
from llm_benchmark.runs.metadata import ReproducibilityRecorder
from llm_benchmark.runs.validator import RunArtifactValidator


class BenchmarkPipeline:
    """Pipeline facade for the currently implemented stages.

    Deployment and user-interface concerns are intentionally separate.
    """

    def __init__(
        self,
        run_manager: RunManager | None = None,
        dataset_service: DatasetIngestionService | None = None,
    ) -> None:
        self._run_manager = run_manager or RunManager()
        self._dataset_service = dataset_service or DatasetIngestionService()

    def initialize_run(
        self, config: AppConfig, config_source: str | Path
    ) -> RunContext:
        return self._run_manager.initialize(config, config_source)

    def open_run(self, config: AppConfig, run_dir: str | Path) -> RunContext:
        return self._run_manager.open_existing(config, run_dir)

    def prepare_dataset(
        self, config: AppConfig, context: RunContext
    ) -> DatasetIngestionResult:
        self._run_manager.assert_can_prepare_dataset(context)
        result = self._dataset_service.ingest(config.dataset, context.run_dir)
        self._run_manager.mark_dataset_prepared(context, result)
        return result

    async def run_inference(
        self,
        config: AppConfig,
        context: RunContext,
        *,
        resume: bool,
        model_client: ModelClient | None = None,
    ) -> InferenceRunResult:
        owned_client = model_client is None
        client = model_client or OpenAICompatibleClient(
            retry_policy=RetryPolicy(
                max_retries=config.runtime.max_retries,
                initial_backoff_seconds=config.runtime.retry_backoff_seconds,
                maximum_backoff_seconds=max(
                    8.0, config.runtime.retry_backoff_seconds
                ),
            )
        )
        self._run_manager.mark_inference_started(context)
        try:
            result = await InferenceEngine(client).run(
                normalized_dataset_path=(
                    context.run_dir / "dataset" / "normalized_dataset.jsonl"
                ),
                run_dir=context.run_dir,
                base_model=config.models.base,
                fine_tuned_model=config.models.fine_tuned,
                concurrency=config.runtime.concurrency,
                resume=resume,
            )
        finally:
            if owned_client:
                await client.aclose()
        self._run_manager.mark_inference_finished(context, result)
        return result

    async def run_evaluation(
        self,
        config: AppConfig,
        context: RunContext,
        *,
        resume: bool,
        model_client: ModelClient | None = None,
    ) -> JudgeRunResult:
        owned_client = model_client is None
        client = model_client or OpenAICompatibleClient(
            retry_policy=RetryPolicy(
                max_retries=config.runtime.max_retries,
                initial_backoff_seconds=config.runtime.retry_backoff_seconds,
                maximum_backoff_seconds=max(
                    8.0, config.runtime.retry_backoff_seconds
                ),
            )
        )
        self._run_manager.mark_evaluation_started(context)
        try:
            result = await JudgeEngine(client).run(
                inference_path=context.run_dir / "inference" / "responses.jsonl",
                run_dir=context.run_dir,
                judge_model=config.models.judge,
                prompt_template_path=config.evaluation.prompt_template,
                criteria=config.evaluation.criteria,
                concurrency=config.runtime.concurrency,
                random_seed=config.runtime.random_seed,
                resume=resume,
            )
        finally:
            if owned_client:
                await client.aclose()
        self._run_manager.mark_evaluation_finished(context, result)
        return result

    def generate_reports(
        self,
        config: AppConfig,
        context: RunContext,
    ) -> ReportGenerationResult:
        manifest = self._run_manager.mark_reporting_started(context)
        judge_results = list(
            JudgeArtifactStore(context.run_dir).load_results().values()
        )
        inference_responses = InferenceResponseReader().read(
            context.run_dir / "inference" / "responses.jsonl"
        )
        analysis = BenchmarkMetricsCalculator().calculate(
            judge_results,
            inference_responses,
        )
        result = ReportGenerator().generate(
            config=config,
            manifest=manifest,
            analysis=analysis,
            run_dir=context.run_dir,
        )
        validation = RunArtifactValidator().validate_and_write(
            config=config,
            context=context,
            manifest=manifest,
            report_result=result,
        )
        reproducibility_path = ReproducibilityRecorder().write(
            config=config,
            context=context,
            manifest=manifest,
            report_result=result,
            validation_path=validation.artifact_path,
        )
        self._run_manager.mark_reporting_finished(
            context,
            result,
            validation_path=validation.artifact_path,
            reproducibility_path=reproducibility_path,
        )
        return result
