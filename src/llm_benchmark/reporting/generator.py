"""Render benchmark analysis as JSON, Markdown, and CSV artifacts."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from llm_benchmark.config.models import AppConfig, EndpointConfig
from llm_benchmark.domain.runs import RunManifest
from llm_benchmark.metrics.models import BenchmarkAnalysis, ResolvedSampleResult
from llm_benchmark.reporting.errors import ReportGenerationError
from llm_benchmark.reporting.models import (
    BenchmarkMetadata,
    BenchmarkReport,
    ComparedModels,
    CriterionInformation,
    DatasetInformation,
    ModelInformation,
    ReportGenerationResult,
)
from llm_benchmark.runs.artifact_store import LocalArtifactStore


_LOGGER = logging.getLogger("llm_benchmark.reporting.generator")


class ReportGenerator:
    JSON_RELATIVE = Path("reports/report.json")
    MARKDOWN_RELATIVE = Path("reports/report.md")
    CSV_RELATIVE = Path("reports/samples.csv")

    def generate(
        self,
        *,
        config: AppConfig,
        manifest: RunManifest,
        analysis: BenchmarkAnalysis,
        run_dir: Path,
    ) -> ReportGenerationResult:
        if manifest.dataset is None:
            raise ReportGenerationError(
                "Run manifest does not contain prepared dataset metadata"
            )

        _LOGGER.info(
            "Report generation started",
            extra={
                "event": "reporting_started",
                "run_id": manifest.run_id,
                "stage": "reporting",
            },
        )

        report = BenchmarkReport(
            comparison_status=(
                "comparison_available"
                if analysis.aggregate_metrics.successful_evaluations > 0
                else "no_successful_evaluations"
            ),
            benchmark_metadata=BenchmarkMetadata(
                run_id=manifest.run_id,
                generated_at=datetime.now(UTC),
                pipeline_version=manifest.pipeline_version,
                config_fingerprint=manifest.config_fingerprint,
            ),
            model_information=ComparedModels(
                base=self._model_information(config.models.base),
                fine_tuned=self._model_information(config.models.fine_tuned),
                judge=self._model_information(config.models.judge),
            ),
            dataset_information=DatasetInformation(
                source_path=manifest.dataset.source_path,
                source_format=manifest.dataset.source_format,
                schema_format=manifest.dataset.schema_format,
                sample_count=manifest.dataset.sample_count,
                source_sha256=manifest.dataset.source_sha256,
                detection_confidence=manifest.dataset.detection_confidence,
                detected_fields=manifest.dataset.detected_fields,
            ),
            evaluation_criteria=[
                CriterionInformation(
                    name=criterion.name,
                    description=criterion.description,
                    weight=criterion.weight,
                    minimum=criterion.minimum,
                    maximum=criterion.maximum,
                )
                for criterion in config.evaluation.criteria
            ],
            aggregate_metrics=analysis.aggregate_metrics,
            per_sample_results=analysis.samples,
        )

        store = LocalArtifactStore(run_dir)
        report_json_path = store.write_json(
            self.JSON_RELATIVE,
            report.model_dump(mode="json"),
        )
        report_markdown_path = store.write_text(
            self.MARKDOWN_RELATIVE,
            self._render_markdown(report),
        )
        samples_csv_path = store.write_text(
            self.CSV_RELATIVE,
            self._render_csv(report.per_sample_results),
        )
        result = ReportGenerationResult(
            report_json_path=report_json_path,
            report_markdown_path=report_markdown_path,
            samples_csv_path=samples_csv_path,
            report=report,
        )
        _LOGGER.info(
            "Report generation completed",
            extra={
                "event": "reporting_completed",
                "run_id": manifest.run_id,
                "stage": "reporting",
            },
        )
        return result

    @staticmethod
    def _model_information(endpoint: EndpointConfig) -> ModelInformation:
        return ModelInformation(
            name=endpoint.name,
            base_url=str(endpoint.base_url),
            timeout_seconds=endpoint.timeout_seconds,
            generation_parameters=dict(endpoint.generation_parameters),
        )

    def _render_markdown(self, report: BenchmarkReport) -> str:
        metrics = report.aggregate_metrics
        rates = metrics.win_rate_percentage
        averages = metrics.average_scores_per_model
        dataset = report.dataset_information
        models = report.model_information
        lines = [
            "# Benchmark Report",
            "",
            "## Benchmark summary",
            "",
            f"- Run ID: `{self._inline(report.benchmark_metadata.run_id)}`",
            f"- Generated: {report.benchmark_metadata.generated_at.isoformat()}",
            f"- Total samples: {metrics.total_samples}",
            f"- Successful evaluations: {metrics.successful_evaluations}",
            f"- Failed evaluations: {metrics.failed_evaluations}",
            f"- Skipped evaluations: {metrics.skipped_evaluations}",
            f"- Comparison status: `{report.comparison_status}`",
            "",
            "## Compared models",
            "",
            "| Role | Model | Endpoint |",
            "|---|---|---|",
            f"| Base | {self._table(models.base.name)} | {self._table(models.base.base_url)} |",
            f"| Fine-tuned | {self._table(models.fine_tuned.name)} | {self._table(models.fine_tuned.base_url)} |",
            f"| Judge | {self._table(models.judge.name)} | {self._table(models.judge.base_url)} |",
            "",
            "## Dataset information",
            "",
            f"- Source: `{self._inline(dataset.source_path)}`",
            f"- File format: {self._inline(dataset.source_format)}",
            f"- Detected schema: {self._inline(dataset.schema_format)}",
            f"- Samples: {dataset.sample_count}",
            f"- SHA-256: `{dataset.source_sha256}`",
            "",
            "## Overall results",
            "",
            "| Outcome | Count | Rate among successful evaluations |",
            "|---|---:|---:|",
            f"| Base model wins | {metrics.base_model_wins} | {rates.base_model:.2f}% |",
            f"| Fine-tuned model wins | {metrics.fine_tuned_model_wins} | {rates.fine_tuned_model:.2f}% |",
            f"| Ties | {metrics.ties} | {rates.ties:.2f}% |",
            "",
            "| Model | Average score |",
            "|---|---:|",
            f"| Base | {self._score(averages.base_model)} |",
            f"| Fine-tuned | {self._score(averages.fine_tuned_model)} |",
            "",
            "## Evaluation criteria",
            "",
            "| Criterion | Weight | Range | Description |",
            "|---|---:|---:|---|",
        ]
        if report.comparison_status == "no_successful_evaluations":
            position = lines.index("## Overall results") + 2
            lines[position:position] = [
                "**No model comparison conclusion is available because zero judge evaluations succeeded.**",
                "",
            ]
        for criterion in report.evaluation_criteria:
            lines.append(
                f"| {self._table(criterion.name)} | {criterion.weight:.4g} | "
                f"{criterion.minimum:g}–{criterion.maximum:g} | "
                f"{self._table(criterion.description)} |"
            )

        lines.extend(["", "## Examples", ""])
        for sample in report.per_sample_results[:5]:
            lines.extend(self._render_example(sample))
        if not report.per_sample_results:
            lines.append("No samples were available.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_example(sample: ResolvedSampleResult) -> list[str]:
        winner = sample.winner.value if sample.winner is not None else "unavailable"
        reason = sample.judge_reason or sample.error_message or "Not available"
        return [
            f"### Sample `{ReportGenerator._inline(sample.id)}`",
            "",
            f"- Evaluation status: {sample.evaluation_status.value}",
            f"- Winner: {winner}",
            f"- Base score: {ReportGenerator._score(sample.base_score)}",
            f"- Fine-tuned score: {ReportGenerator._score(sample.fine_tuned_score)}",
            "",
            "**Prompt**",
            "",
            ReportGenerator._quote(sample.prompt),
            "",
            "**Base response**",
            "",
            ReportGenerator._quote(sample.base_response),
            "",
            "**Fine-tuned response**",
            "",
            ReportGenerator._quote(sample.fine_tuned_response),
            "",
            "**Judge reason**",
            "",
            ReportGenerator._quote(reason),
            "",
        ]

    @staticmethod
    def _render_csv(samples: list[ResolvedSampleResult]) -> str:
        output = io.StringIO(newline="")
        fieldnames = [
            "sample_id",
            "prompt",
            "base_response",
            "fine_tuned_response",
            "evaluation_status",
            "winner",
            "base_score",
            "fine_tuned_score",
            "criteria_scores",
            "judge_reason",
            "error_message",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": ReportGenerator._csv_cell(sample.id),
                    "prompt": ReportGenerator._csv_cell(sample.prompt),
                    "base_response": ReportGenerator._csv_cell(
                        sample.base_response
                    ),
                    "fine_tuned_response": ReportGenerator._csv_cell(
                        sample.fine_tuned_response
                    ),
                    "evaluation_status": sample.evaluation_status.value,
                    "winner": sample.winner.value if sample.winner else "",
                    "base_score": (
                        sample.base_score if sample.base_score is not None else ""
                    ),
                    "fine_tuned_score": (
                        sample.fine_tuned_score
                        if sample.fine_tuned_score is not None
                        else ""
                    ),
                    "criteria_scores": (
                        json.dumps(
                            {
                                key: value.model_dump(mode="json")
                                for key, value in sample.criteria_scores.items()
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        if sample.criteria_scores is not None
                        else ""
                    ),
                    "judge_reason": ReportGenerator._csv_cell(
                        sample.judge_reason or ""
                    ),
                    "error_message": ReportGenerator._csv_cell(
                        sample.error_message or ""
                    ),
                }
            )
        return output.getvalue()

    @staticmethod
    def _score(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4g}"

    @staticmethod
    def _table(value: str) -> str:
        return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _inline(value: str) -> str:
        return value.replace("`", "\\`").replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _quote(value: str, limit: int = 1000) -> str:
        shortened = value if len(value) <= limit else value[:limit] + "…"
        lines = shortened.splitlines() or [""]
        return "\n".join(f"> {line}" for line in lines)

    @staticmethod
    def _csv_cell(value: str) -> str:
        if value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
