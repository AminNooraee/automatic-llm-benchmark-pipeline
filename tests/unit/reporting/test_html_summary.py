from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from llm_benchmark.evaluation.models import (
    AnswerOrder,
    EvaluatedModel,
    JudgeEvaluationStatus,
    JudgeWinner,
    ResolvedWinner,
)
from llm_benchmark.metrics.models import (
    AggregateMetrics,
    ModelAverageScores,
    ModelWinRates,
    ResolvedCriteriaScores,
    ResolvedSampleResult,
)
from llm_benchmark.reporting.cost import estimate_reference_costs, load_pricing
from llm_benchmark.reporting.html_summary import SummaryData, render_benchmark_summary_html
from llm_benchmark.reporting.models import (
    BenchmarkMetadata,
    BenchmarkReport,
    ComparedModels,
    CriterionInformation,
    DatasetInformation,
    ModelInformation,
)
from llm_benchmark.reporting.token_usage import RoleUsage, TokenUsageSummary

ROOT = Path(__file__).resolve().parents[3]


def _model(name: str) -> ModelInformation:
    return ModelInformation(
        name=name,
        base_url="https://example.invalid/v1",
        timeout_seconds=10,
        generation_parameters={},
    )


def _sample(index: int, winner: ResolvedWinner) -> ResolvedSampleResult:
    judge_winner = JudgeWinner.TIE if winner == ResolvedWinner.TIE else JudgeWinner.A
    return ResolvedSampleResult(
        id=f"sample-{index}",
        prompt=f'<script>alert("prompt-{index}")</script>',
        base_response=f"پاسخ پایه {index} & more",
        fine_tuned_response=f"fine answer {index} <b>unsafe</b>",
        evaluation_status=JudgeEvaluationStatus.SUCCESS,
        winner=winner,
        judge_winner_label=judge_winner,
        answer_order=AnswerOrder(A=EvaluatedModel.FINE_TUNED, B=EvaluatedModel.BASE),
        base_score=2 if winner == ResolvedWinner.FINE_TUNED else 5,
        fine_tuned_score=5 if winner == ResolvedWinner.FINE_TUNED else 2,
        criteria_scores={
            "correctness <unsafe>": ResolvedCriteriaScores(
                base_model=2,
                fine_tuned_model=5,
            )
        },
        judge_reason=f"دلیل {index} <img src=x onerror=alert(1)>",
    )


def _summary_data() -> SummaryData:
    samples = (
        [_sample(i, ResolvedWinner.FINE_TUNED) for i in range(1, 5)]
        + [_sample(i, ResolvedWinner.BASE) for i in range(5, 8)]
        + [_sample(8, ResolvedWinner.TIE)]
    )
    report = BenchmarkReport(
        comparison_status="comparison_available",
        benchmark_metadata=BenchmarkMetadata(
            run_id="run-technical-id",
            generated_at=datetime(2026, 10, 17, tzinfo=UTC),
            pipeline_version="1.0.0",
            config_fingerprint="abc123",
        ),
        model_information=ComparedModels(
            base=_model("base/model"),
            fine_tuned=_model("fine/model"),
            judge=_model("qwen38-sglang"),
        ),
        dataset_information=DatasetInformation(
            source_path="dataset.jsonl",
            source_format="jsonl",
            schema_format="prompt",
            sample_count=8,
            source_sha256="deadbeef",
            detection_confidence=1,
            detected_fields=["prompt"],
        ),
        evaluation_criteria=[
            CriterionInformation(
                name="correctness <unsafe>",
                description="توضیح & description",
                weight=1,
                minimum=1,
                maximum=5,
            )
        ],
        aggregate_metrics=AggregateMetrics(
            total_samples=8,
            successful_evaluations=8,
            failed_evaluations=0,
            skipped_evaluations=0,
            base_model_wins=3,
            fine_tuned_model_wins=4,
            ties=1,
            win_rate_percentage=ModelWinRates(
                base_model=37.5,
                fine_tuned_model=50,
                ties=12.5,
            ),
            average_scores_per_model=ModelAverageScores(
                base_model=3,
                fine_tuned_model=4,
            ),
        ),
        per_sample_results=samples,
    )
    usage = RoleUsage(
        expected_successful=1,
        reported=1,
        available=1,
        prompt_tokens=2321,
        completion_tokens=590,
        total_tokens=2911,
    )
    return SummaryData(
        report=report,
        validation_status="passed",
        random_seed=7,
        pipeline_version="1.0.0",
        run_id="run-technical-id",
        generated_at=datetime(2026, 10, 17, tzinfo=UTC),
        training_duration_seconds=None,
        inference_duration_seconds=20,
        judge_duration_seconds=30,
        report_context=None,
        token_summary=TokenUsageSummary(base=usage, fine_tuned=usage, judge=usage),
        actual_judge_name="qwen38-sglang",
        judge_costs=estimate_reference_costs(
            judge_usage=usage,
            pricing=load_pricing(ROOT / "configs" / "judge_pricing.yaml"),
        ),
    )


def test_shared_renderer_produces_self_contained_bilingual_dashboards() -> None:
    data = _summary_data()
    persian = render_benchmark_summary_html(data, locale="fa")
    english = render_benchmark_summary_html(data, locale="en")

    assert '<html lang="fa" dir="rtl">' in persian
    assert '<html lang="en" dir="ltr">' in english
    assert "خلاصهٔ نتیجه" in persian
    assert "Executive Summary" in english
    assert "خلاصهٔ نتیجه" not in english
    assert "مدل داور واقعی" not in english
    assert "Actual Judge Model" in english
    for rendered in (persian, english):
        assert '<meta charset="UTF-8">' in rendered
        assert rendered.count("2026-10-14") == 1
        assert rendered.count('class="sample-card"') == 6
        assert rendered.count('class="answer" dir="auto"') == 12
        assert '<span class="ltr" dir="ltr">run-technical-id</span>' in rendered
        assert "&lt;script&gt;" in rendered
        assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
        assert "<script>" not in rendered
        assert "<link " not in rendered
        assert "@import" not in rendered


def test_actual_judge_and_all_reference_costs_are_rendered_once_per_locale() -> None:
    data = _summary_data()
    for locale in ("fa", "en"):
        rendered = render_benchmark_summary_html(data, locale=locale)
        assert rendered.count("qwen38-sglang") == 1
        assert rendered.count("OpenAI GPT-5.6 Sol") == 1
        assert rendered.count("Anthropic Claude Sonnet 5.5") == 1
        assert rendered.count("DeepSeek V4-Pro") == 1
        assert "0.021084 USD" in rendered
        assert "0.010542 USD" in rendered
        assert "0.002700 – 0.005400 USD" in rendered
        assert rendered.count("2,321") == 3
        assert rendered.count("590") == 3
        assert rendered.count("2,911") == 3


def test_incomplete_judge_usage_keeps_model_and_makes_all_costs_unavailable() -> None:
    data = _summary_data()
    incomplete = RoleUsage(
        expected_successful=2,
        reported=1,
        available=1,
        prompt_tokens=2321,
        completion_tokens=590,
        total_tokens=2911,
    )
    data = replace(
        data,
        token_summary=replace(data.token_summary, judge=incomplete),
        judge_costs=estimate_reference_costs(
            judge_usage=incomplete,
            pricing=load_pricing(ROOT / "configs" / "judge_pricing.yaml"),
        ),
    )
    rendered = render_benchmark_summary_html(data, locale="en")

    assert "qwen38-sglang" in rendered
    assert rendered.count('<span class="missing">Unavailable</span>') >= 3
    assert "0.021084 USD" not in rendered
    assert "0.010542 USD" not in rendered
    assert "0.002700 – 0.005400 USD" not in rendered
