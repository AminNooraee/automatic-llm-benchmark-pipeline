from __future__ import annotations

from datetime import UTC, datetime

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
    ResolvedSampleResult,
)
from llm_benchmark.reporting.cost import JudgeCost
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
        criteria_scores=None,
        judge_reason=f"دلیل {index} <img src=x onerror=alert(1)>",
    )


def _summary_data(*, with_context=False) -> SummaryData:
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
            judge=_model("gpt-5.6-sol"),
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
                base_model=37.5, fine_tuned_model=50, ties=12.5
            ),
            average_scores_per_model=ModelAverageScores(base_model=3, fine_tuned_model=4),
        ),
        per_sample_results=samples,
    )
    role = RoleUsage(
        expected_successful=1,
        reported=1,
        available=1,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    return SummaryData(
        report=report,
        validation_status="passed",
        random_seed=7,
        pipeline_version="1.0.0",
        run_id="run-technical-id",
        generated_at=datetime(2026, 10, 17, tzinfo=UTC),
        training_duration_seconds=10 if with_context else None,
        inference_duration_seconds=20,
        judge_duration_seconds=30,
        report_context=None,
        token_summary=TokenUsageSummary(base=role, fine_tuned=role, judge=role),
        judge_cost=JudgeCost(
            priced=True,
            model_display="OpenAI GPT-5.6 Sol",
            currency="USD",
            snapshot_date="2026-10-14",
            kind="flat",
            cost=0.0008,
        ),
    )


def test_html_is_self_contained_rtl_safe_and_concise() -> None:
    rendered = render_benchmark_summary_html(_summary_data())

    assert '<html lang="fa" dir="rtl">' in rendered
    assert '<meta charset="UTF-8">' in rendered
    assert rendered.count("2026-10-14") == 1
    assert "مبنای قیمت‌گذاری:" in rendered
    assert rendered.count("<details>") == 6
    assert rendered.count('class="answer" dir="auto"') == 12
    assert rendered.count('class="user-content" dir="auto"') >= 12
    assert '<span class="ltr" dir="ltr">run-technical-id</span>' in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<script>" not in rendered
    assert "<link " not in rendered
    assert "@import" not in rendered
    assert "Inter" not in rendered
    assert "ناموجود" in rendered

    section_titles = (
        "خلاصهٔ نتیجه",
        "زمان اجرا",
        "تنظیمات آموزش",
        "مصرف توکن و هزینهٔ تخمینی داوری",
        "معیارهای ارزیابی",
        "چند نمونهٔ منتخب",
        "اعتبارسنجی و بازتولیدپذیری",
    )
    for title in section_titles:
        assert rendered.count(title) == 1


def test_missing_timing_and_incomplete_judge_cost_render_unavailable() -> None:
    data = _summary_data()
    incomplete = RoleUsage(
        expected_successful=2,
        reported=1,
        available=1,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    data = SummaryData(
        **{
            **data.__dict__,
            "token_summary": TokenUsageSummary(
                base=data.token_summary.base,
                fine_tuned=data.token_summary.fine_tuned,
                judge=incomplete,
            ),
            "judge_cost": JudgeCost(
                priced=False,
                model_display="OpenAI GPT-5.6 Sol",
                currency="USD",
                snapshot_date="2026-10-14",
                kind="none",
            ),
        }
    )
    rendered = render_benchmark_summary_html(data)
    assert "ناقص" in rendered
    assert "0.0008 USD" not in rendered
    assert rendered.count("2026-10-14") == 1

