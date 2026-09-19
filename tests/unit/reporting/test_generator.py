from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from llm_benchmark.config.loader import load_config
from llm_benchmark.domain.runs import (
    DatasetManifest,
    RunManifest,
    RunStatus,
)
from llm_benchmark.evaluation.models import (
    AnswerOrder,
    EvaluatedModel,
    JudgeEvaluationStatus,
    ResolvedWinner,
)
from llm_benchmark.metrics.models import (
    AggregateMetrics,
    BenchmarkAnalysis,
    ModelAverageScores,
    ModelWinRates,
    ResolvedSampleResult,
)
from llm_benchmark.reporting.generator import ReportGenerator


def test_report_generator_creates_json_markdown_and_csv(
    config_factory,
    tmp_path: Path,
) -> None:
    config = load_config(config_factory())
    now = datetime.now(UTC)
    manifest = RunManifest(
        run_id="report-test",
        status=RunStatus.EVALUATION_COMPLETED,
        created_at=now,
        updated_at=now,
        pipeline_version="1.0.0",
        config_source=str(tmp_path / "config.yaml"),
        config_fingerprint=config.fingerprint(),
        artifacts={},
        dataset=DatasetManifest(
            source_path=str(config.dataset.path),
            source_format="jsonl",
            schema_format="prompt",
            detection_confidence=1,
            detected_fields=["id", "prompt"],
            sample_count=1,
            source_sha256="a" * 64,
            original_artifact="dataset/original_dataset.jsonl",
            normalized_artifact="dataset/normalized_dataset.jsonl",
        ),
    )
    sample = ResolvedSampleResult(
        id="one",
        prompt="Explain AI",
        base_response="Base answer",
        fine_tuned_response="Fine answer",
        evaluation_status=JudgeEvaluationStatus.SUCCESS,
        winner=ResolvedWinner.FINE_TUNED,
        judge_winner_label="B",
        answer_order=AnswerOrder(
            A=EvaluatedModel.BASE,
            B=EvaluatedModel.FINE_TUNED,
        ),
        base_score=3,
        fine_tuned_score=5,
        criteria_scores={},
        judge_reason="The fine-tuned response is clearer.",
    )
    analysis = BenchmarkAnalysis(
        aggregate_metrics=AggregateMetrics(
            total_samples=1,
            successful_evaluations=1,
            failed_evaluations=0,
            skipped_evaluations=0,
            base_model_wins=0,
            fine_tuned_model_wins=1,
            ties=0,
            win_rate_percentage=ModelWinRates(
                base_model=0,
                fine_tuned_model=100,
                ties=0,
            ),
            average_scores_per_model=ModelAverageScores(
                base_model=3,
                fine_tuned_model=5,
            ),
        ),
        samples=[sample],
    )

    generated = ReportGenerator().generate(
        config=config,
        manifest=manifest,
        analysis=analysis,
        run_dir=tmp_path,
    )

    payload = json.loads(generated.report_json_path.read_text(encoding="utf-8"))
    assert payload["benchmark_metadata"]["run_id"] == "report-test"
    assert payload["model_information"]["base"]["name"] == "base-model"
    assert payload["dataset_information"]["sample_count"] == 1
    assert payload["evaluation_criteria"][0]["name"] == "correctness"
    assert payload["aggregate_metrics"]["fine_tuned_model_wins"] == 1
    assert payload["per_sample_results"][0]["fine_tuned_response"] == "Fine answer"

    markdown = generated.report_markdown_path.read_text(encoding="utf-8")
    assert "# Benchmark Report" in markdown
    assert "## Overall results" in markdown
    assert "Fine-tuned model wins | 1 | 100.00%" in markdown
    assert "## Examples" in markdown

    with generated.samples_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "one"
    assert rows[0]["base_response"] == "Base answer"
    assert rows[0]["fine_tuned_response"] == "Fine answer"
    assert rows[0]["winner"] == "fine_tuned"
    assert rows[0]["base_score"] == "3.0"
    assert rows[0]["fine_tuned_score"] == "5.0"


def test_csv_cells_neutralize_spreadsheet_formulas() -> None:
    assert ReportGenerator._csv_cell("=HYPERLINK(\"bad\")") == (
        "'=HYPERLINK(\"bad\")"
    )
    assert ReportGenerator._csv_cell("ordinary text") == "ordinary text"
