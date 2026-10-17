"""Top-level application orchestration for implemented phases."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from llm_benchmark.clients.contracts import RetryPolicy
from llm_benchmark.clients.interface import ModelClient
from llm_benchmark.exceptions import BenchmarkError
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.models import AppConfig
from llm_benchmark.datasets.models import DatasetIngestionResult
from llm_benchmark.datasets.service import DatasetIngestionService
from llm_benchmark.domain.runs import RunContext, RunManifest
from llm_benchmark.evaluation.engine import JudgeEngine
from llm_benchmark.evaluation.artifact_store import JudgeArtifactStore
from llm_benchmark.evaluation.models import JudgeRunResult
from llm_benchmark.evaluation.reader import InferenceResponseReader
from llm_benchmark.inference.engine import InferenceEngine
from llm_benchmark.inference.models import InferenceRunResult
from llm_benchmark.metrics.calculator import BenchmarkMetricsCalculator
from llm_benchmark.reporting.cost import (
    JudgeCostComparison,
    estimate_reference_costs,
    load_pricing,
)
from llm_benchmark.reporting.generator import ReportGenerator
from llm_benchmark.reporting.html_summary import (
    SummaryData,
    render_benchmark_summary_html,
)
from llm_benchmark.reporting.models import ReportGenerationResult
from llm_benchmark.reporting.token_usage import summarize_token_usage
from llm_benchmark.preflight import EndpointPreflight
from llm_benchmark.inference.artifact_store import InferenceArtifactStore
from llm_benchmark.runs.artifact_store import LocalArtifactStore
from llm_benchmark.runs.manager import RunManager
from llm_benchmark.runs.metadata import ReproducibilityRecorder
from llm_benchmark.runs.validator import RunArtifactValidator

if TYPE_CHECKING:
    from llm_benchmark.config.handoff import ReportContext


class BenchmarkPipeline:
    """Pipeline facade for the currently implemented stages.

    Deployment and user-interface concerns are intentionally separate.
    """

    def __init__(
        self,
        run_manager: RunManager | None = None,
        dataset_service: DatasetIngestionService | None = None,
        report_context: "ReportContext | None" = None,
    ) -> None:
        self._run_manager = run_manager or RunManager()
        self._dataset_service = dataset_service or DatasetIngestionService()
        self._report_context = report_context
        self._inference_duration: float | None = None
        self._judge_duration: float | None = None

    def initialize_run(
        self, config: AppConfig, config_source: str | Path
    ) -> RunContext:
        return self._run_manager.initialize(config, config_source)

    def open_run(self, config: AppConfig, run_dir: str | Path) -> RunContext:
        return self._run_manager.open_existing(config, run_dir)

    async def preflight(
        self, config: AppConfig, *, model_client: ModelClient | None = None
    ) -> None:
        await EndpointPreflight().run(config, model_client=model_client)

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
            inference_started = time.perf_counter()
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
            self._inference_duration = time.perf_counter() - inference_started
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
            judge_started = time.perf_counter()
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
            self._judge_duration = time.perf_counter() - judge_started
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
        self._write_html_summary(config, context, manifest, result, validation.summary.status)
        self._run_manager.mark_reporting_finished(
            context,
            result,
            validation_path=validation.artifact_path,
            reproducibility_path=reproducibility_path,
        )
        return result

    def _write_html_summary(
        self,
        config: AppConfig,
        context: RunContext,
        manifest: RunManifest,
        result: ReportGenerationResult,
        validation_status: str,
    ) -> None:
        """Render the self-contained Persian and English summary dashboards.

        This artifact is additive: it never modifies or invalidates the existing
        report artifacts, and it degrades gracefully when the optional report
        context, stage durations, token usage, or pricing data are unavailable.
        """

        try:
            inference_responses = InferenceArtifactStore(
                context.run_dir
            ).read_responses()
            judge_results = list(
                JudgeArtifactStore(context.run_dir).load_results().values()
            )
        except BenchmarkError:
            # Token usage is optional for the summary: if the inference or
            # judge artifacts cannot be read, degrade to all-missing usage
            # rather than failing the additive HTML artifact.
            inference_responses = []
            judge_results = []
        token_summary = summarize_token_usage(inference_responses, judge_results)

        judge_costs = _empty_judge_costs()
        pricing = _load_pricing_config(_config_dir(manifest))
        if pricing is not None:
            judge_costs = estimate_reference_costs(
                judge_usage=token_summary.judge,
                pricing=pricing,
            )

        training_duration = (
            self._report_context.training_duration_seconds
            if self._report_context is not None
            else None
        )
        data = SummaryData(
            report=result.report,
            validation_status=validation_status,
            random_seed=config.runtime.random_seed,
            pipeline_version=result.report.benchmark_metadata.pipeline_version,
            run_id=context.run_id,
            generated_at=result.report.benchmark_metadata.generated_at,
            training_duration_seconds=training_duration,
            inference_duration_seconds=self._inference_duration,
            judge_duration_seconds=self._judge_duration,
            report_context=self._report_context,
            token_summary=token_summary,
            actual_judge_name=config.models.judge.name,
            judge_costs=judge_costs,
        )
        artifact_store = LocalArtifactStore(context.run_dir)
        artifact_store.write_text(
            "reports/benchmark_summary.html",
            render_benchmark_summary_html(data, locale="fa"),
        )
        artifact_store.write_text(
            "reports/benchmark_summary_en.html",
            render_benchmark_summary_html(data, locale="en"),
        )


def _config_dir(manifest: RunManifest) -> Path:
    """The directory that contains the benchmark config that started the run."""

    return Path(manifest.config_source).parent


def _empty_judge_costs() -> JudgeCostComparison:
    return JudgeCostComparison(
        currency="USD",
        snapshot_date="",
        estimates=(),
    )


def _load_pricing_config(config_dir: Path) -> dict | None:
    """Load the judge pricing snapshot next to the benchmark config, if present.

    Missing or unreadable pricing never fails the run; it simply means no cost
    can be estimated (rendered as ناموجود).
    """

    candidate = config_dir / "judge_pricing.yaml"
    if not candidate.is_file():
        return None
    try:
        return load_pricing(candidate)
    except (OSError, ValueError, yaml.YAMLError):
        return None
