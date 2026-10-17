"""Self-contained Persian RTL benchmark summary page.

Renders a single static ``benchmark_summary.html`` that aggregates the
benchmark report, validation/reproducibility metadata, training report context,
stage durations, per-role token usage, and an estimated judge API cost.

The page is fully self-contained: no CDN, no external JavaScript or fonts, no
network access. All dataset and user content is HTML-escaped; technical values
(model names, run IDs, hashes, paths, numbers) are rendered left-to-right.
Missing data is rendered as ``ناموجود`` and incomplete token coverage as
``ناقص`` rather than as misleading zeros.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime

from llm_benchmark.config.handoff import ReportContext
from llm_benchmark.metrics.models import ResolvedSampleResult
from llm_benchmark.reporting.cost import JudgeCost
from llm_benchmark.reporting.models import BenchmarkReport
from llm_benchmark.reporting.token_usage import TokenUsageSummary

MISSING = "ناموجود"
INCOMPLETE = "ناقص"

_PERSIAN_WINNERS = {
    "base": "مدل پایه",
    "fine_tuned": "مدل فاین‌تیون‌شده",
    "tie": "مساوی",
}


@dataclass(frozen=True)
class SummaryData:
    """Everything needed to render the benchmark summary page."""

    report: BenchmarkReport
    validation_status: str
    random_seed: int
    pipeline_version: str
    run_id: str
    generated_at: datetime
    training_duration_seconds: float | None
    inference_duration_seconds: float | None
    judge_duration_seconds: float | None
    report_context: ReportContext | None
    token_summary: TokenUsageSummary
    judge_cost: JudgeCost


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _ltr(value: object) -> str:
    """Render a technical value in a left-to-right inline run."""

    return f'<span class="ltr" dir="ltr">{_esc(value)}</span>'


def _auto(value: object, css_class: str = "user-content") -> str:
    """Render user/dataset-authored text with automatic bidi detection."""

    return f'<span class="{css_class}" dir="auto">{_esc(value)}</span>'


def _missing_or(value: object, fallback: str = MISSING) -> str:
    if value is None:
        return f'<span class="missing">{fallback}</span>'
    return str(value)


def _fmt_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return f'<span class="missing">{MISSING}</span>'
    text = f"{value:.{digits}g}"
    return _ltr(text)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return f'<span class="missing">{MISSING}</span>'
    total = max(0.0, float(seconds))
    minutes, secs = divmod(round(total), 60)
    if minutes:
        return f'{_ltr(f"{minutes} دقیقه و {secs} ثانیه")}'
    return f'{_ltr(f"{secs} ثانیه")}'


def _fmt_tokens(
    usage,
    field: str,
) -> str:
    """Render one token field, honoring complete / incomplete / missing."""

    if usage.missing:
        return f'<span class="missing">{MISSING}</span>'
    if not usage.complete:
        return f'<span class="incomplete">{INCOMPLETE}</span>'
    value = getattr(usage, field)
    if value is None:
        return f'<span class="missing">{MISSING}</span>'
    return _ltr(value)


def _fmt_cost(cost: JudgeCost) -> str:
    if not cost.priced:
        return f'<span class="missing">{MISSING}</span>'
    if cost.kind == "flat" and cost.cost is not None:
        return _ltr(f"{cost.cost:.4f} {cost.currency}")
    if cost.kind == "range" and cost.cost_low is not None and cost.cost_high is not None:
        return _ltr(f"{cost.cost_low:.4f} – {cost.cost_high:.4f} {cost.currency}")
    return f'<span class="missing">{MISSING}</span>'


def _conclusion(report: BenchmarkReport) -> str:
    metrics = report.aggregate_metrics
    if metrics.successful_evaluations == 0:
        return "به دلیل نبود هیچ ارزیابی موفق، نتیجه‌گیری مقایسه‌ای ممکن نیست."
    base_wins = metrics.base_model_wins
    fine_wins = metrics.fine_tuned_model_wins
    if fine_wins > base_wins:
        return (
            "مدل فاین‌تیون‌شده در این بنچمارک نسبت به مدل پایه عملکرد بهتری "
            "نشان داد."
        )
    if base_wins > fine_wins:
        return "مدل پایه در این بنچمارک نسبت به مدل فاین‌تیون‌شده عملکرد بهتری نشان داد."
    return "عملکرد دو مدل در این بنچمارک تقریباً هم‌ارز بود."


def _winner_label(resolved_value: str | None) -> str:
    if resolved_value is None:
        return f'<span class="missing">{MISSING}</span>'
    return _PERSIAN_WINNERS.get(resolved_value, _esc(resolved_value))


def _score_or(value: float | None) -> str:
    if value is None:
        return f'<span class="missing">{MISSING}</span>'
    return _ltr(f"{value:.4g}")


def _render_metric_row(label: str, value_html: str) -> str:
    return f'<tr><th>{_esc(label)}</th><td>{value_html}</td></tr>'


def _render_samples(samples: list[ResolvedSampleResult]) -> list[str]:
    """Render a group of collapsible samples, ordered as given."""

    lines: list[str] = []
    for sample in samples:
        winner = _winner_label(sample.winner.value if sample.winner else None)
        base_score = _score_or(sample.base_score)
        fine_score = _score_or(sample.fine_tuned_score)
        summary = (
            f'نمونه «{_ltr(sample.id)}» — برنده: {winner} — '
            f'امتیاز پایه: {base_score}، فاین‌تیون‌شده: {fine_score}'
        )
        reason = sample.judge_reason or sample.error_message or MISSING
        lines.append(f"<details>")
        lines.append(f'  <summary>{summary}</summary>')
        lines.append('  <div class="sample-body">')
        lines.append(f'    <p><strong>سؤال:</strong><br>{_auto(sample.prompt)}</p>')
        lines.append(
            f'    <p><strong>پاسخ مدل پایه:</strong><br>'
            f"{_auto(sample.base_response, 'answer')}</p>"
        )
        lines.append(
            f'    <p><strong>پاسخ مدل فاین‌تیون‌شده:</strong><br>'
            f"{_auto(sample.fine_tuned_response, 'answer')}</p>"
        )
        lines.append(
            f'    <p><strong>امتیاز پایه / فاین‌تیون‌شده:</strong> '
            f"{base_score} / {fine_score}</p>"
        )
        lines.append(f'    <p><strong>دلیل Judge:</strong><br>{_auto(reason)}</p>')
        lines.append("  </div>")
        lines.append("</details>")
    return lines


def _select_samples(
    samples: list[ResolvedSampleResult],
) -> tuple[list[ResolvedSampleResult], list[ResolvedSampleResult], ResolvedSampleResult | None]:
    successful = [
        s
        for s in samples
        if s.evaluation_status.value == "success" and s.winner is not None
    ]

    def _gap(sample: ResolvedSampleResult, model_wins: bool) -> float:
        if model_wins:
            return (sample.fine_tuned_score or 0) - (sample.base_score or 0)
        return (sample.base_score or 0) - (sample.fine_tuned_score or 0)

    fine_wins = [
        s for s in successful if s.winner.value == "fine_tuned"
    ]
    base_wins = [s for s in successful if s.winner.value == "base"]
    ties = [s for s in successful if s.winner.value == "tie"]

    fine_wins.sort(key=lambda s: _gap(s, True), reverse=True)
    base_wins.sort(key=lambda s: _gap(s, False), reverse=True)

    return fine_wins[:3], base_wins[:2], (ties[0] if ties else None)


def _render_criteria(report: BenchmarkReport) -> list[str]:
    lines = [
        '<table class="table criteria">',
        "<thead><tr><th>معیار</th><th>وزن</th><th>محدوده</th><th>توضیح</th></tr></thead>",
        "<tbody>",
    ]
    for criterion in report.evaluation_criteria:
        lines.append(
            "<tr>"
            f"<td>{_auto(criterion.name)}</td>"
            f"<td>{_ltr(f'{criterion.weight:.4g}')}</td>"
            f"<td>{_ltr(f'{criterion.minimum:g}–{criterion.maximum:g}')}</td>"
            f"<td>{_auto(criterion.description)}</td>"
            "</tr>"
        )
    lines.append("</tbody>")
    lines.append("</table>")
    return lines


def _render_token_table(summary: TokenUsageSummary) -> list[str]:
    lines = [
        '<table class="table tokens">',
        "<thead><tr><th>نقش</th><th>توکن ورودی</th>"
        "<th>توکن خروجی</th><th>مجموع توکن</th><th>پوشش</th></tr></thead>",
        "<tbody>",
    ]
    rows = (
        ("مدل پایه", summary.base),
        ("مدل فاین‌تیون‌شده", summary.fine_tuned),
        ("داور", summary.judge),
    )
    for label, usage in rows:
        coverage = _coverage_label(usage)
        lines.append(
            "<tr>"
            f"<td>{_esc(label)}</td>"
            f"<td>{_fmt_tokens(usage, 'prompt_tokens')}</td>"
            f"<td>{_fmt_tokens(usage, 'completion_tokens')}</td>"
            f"<td>{_fmt_tokens(usage, 'total_tokens')}</td>"
            f"<td>{coverage}</td>"
            "</tr>"
        )
    lines.append("</tbody>")
    lines.append("</table>")
    return lines


def _coverage_label(usage) -> str:
    if usage.missing:
        return f'<span class="missing">{MISSING}</span>'
    if usage.complete:
        return _ltr(f"{usage.available}/{usage.expected_successful}")
    return (
        f'<span class="incomplete">{INCOMPLETE} '
        f"{_ltr(f'{usage.available}/{usage.expected_successful}')}</span>"
    )


def _render_training_context(context: ReportContext | None) -> list[str]:
    lines = ['<table class="table config">']
    if context is None:
        rows = (
            ("نام مدل پایه", MISSING),
            ("تعداد اپاکش (epochs)", MISSING),
            ("نرخ یادگیری (learning rate)", MISSING),
            ("اندازهٔ دسته (batch size)", MISSING),
            ("طول برش (cutoff length)", MISSING),
            ("رتبهٔ LoRA (rank)", MISSING),
            ("ضریب LoRA (alpha)", MISSING),
            ("ریزش LoRA (dropout)", MISSING),
        )
        for label, value in rows:
            lines.append(
                _render_metric_row(label, f'<span class="missing">{MISSING}</span>')
            )
        lines.append("</table>")
        return lines

    training = context.training
    lora = training.lora
    rows = (
        ("نام مدل پایه", _ltr(training.model_name)),
        ("تعداد اپاکش (epochs)", _ltr(training.epochs)),
        ("نرخ یادگیری (learning rate)", _ltr(training.learning_rate)),
        ("اندازهٔ دسته (batch size)", _ltr(training.batch_size)),
        ("طول برش (cutoff length)", _ltr(training.cutoff_len)),
        (
            "رتبهٔ LoRA (rank)",
            _ltr(lora.rank) if lora else f'<span class="missing">{MISSING}</span>',
        ),
        (
            "ضریب LoRA (alpha)",
            _ltr(lora.alpha) if lora else f'<span class="missing">{MISSING}</span>',
        ),
        (
            "ریزش LoRA (dropout)",
            _ltr(lora.dropout) if lora else f'<span class="missing">{MISSING}</span>',
        ),
    )
    for label, value in rows:
        lines.append(_render_metric_row(label, value))
    lines.append("</table>")
    return lines


def render_benchmark_summary_html(data: SummaryData) -> str:
    report = data.report
    metrics = report.aggregate_metrics
    averages = metrics.average_scores_per_model
    rates = metrics.win_rate_percentage
    usage = data.token_summary
    fine_wins, base_wins, tie = _select_samples(report.per_sample_results)

    score_diff_html = _score_or(
        (averages.fine_tuned_model - averages.base_model)
        if averages.base_model is not None
        and averages.fine_tuned_model is not None
        else None
    )

    total_core = None
    if (
        data.training_duration_seconds is not None
        and data.inference_duration_seconds is not None
        and data.judge_duration_seconds is not None
    ):
        total_core = (
            data.training_duration_seconds
            + data.inference_duration_seconds
            + data.judge_duration_seconds
        )

    cost = data.judge_cost
    cost_model = cost.model_display or MISSING

    token_warning = ""
    if usage.any_incomplete:
        token_warning = (
            '<p class="note">توجه: میزان مصرف توکن برای همهٔ درخواست‌های موفق '
            "در دسترس نبود؛ اعداد نشان‌داده‌شده ممکن است کامل نباشند.</p>"
        )

    cost_note = (
        '<p class="note">هزینهٔ برآوردی فقط بر اساس مصرف توکن گزارش‌شده توسط داور '
        "محاسبه شده و ممکن است با روش شمارش توکن ارائه‌دهنده متفاوت باشد.</p>"
    )

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="fa" dir="rtl">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append(
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
    )
    parts.append("<title>خلاصهٔ بنچمارک</title>")
    parts.append("<style>")
    parts.append(_CSS)
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")
    parts.append("<h1>خلاصهٔ بنچمارک مدل</h1>")

    # 1 — result summary
    parts.append('<section id="summary">')
    parts.append("<h2>۱. خلاصهٔ نتیجه</h2>")
    parts.append('<table class="table">')
    parts.append(_render_metric_row("مجموع نمونه‌ها", _ltr(metrics.total_samples)))
    parts.append(
        _render_metric_row(
            "پیروزی مدل فاین‌تیون‌شده",
            f"{_ltr(metrics.fine_tuned_model_wins)} ({_ltr(f'{rates.fine_tuned_model:.2f}%')})",
        )
    )
    parts.append(
        _render_metric_row(
            "پیروزی مدل پایه",
            f"{_ltr(metrics.base_model_wins)} ({_ltr(f'{rates.base_model:.2f}%')})",
        )
    )
    parts.append(
        _render_metric_row(
            "مساوی",
            f"{_ltr(metrics.ties)} ({_ltr(f'{rates.ties:.2f}%')})",
        )
    )
    parts.append(
        _render_metric_row("میانگین امتیاز مدل پایه", _fmt_number(averages.base_model))
    )
    parts.append(
        _render_metric_row(
            "میانگین امتیاز مدل فاین‌تیون‌شده",
            _fmt_number(averages.fine_tuned_model),
        )
    )
    parts.append(_render_metric_row("اختلاف امتیاز (فاین‌تیون‌شده − پایه)", score_diff_html))
    parts.append("</table>")
    parts.append(f'<p class="conclusion">{_esc(_conclusion(report))}</p>')
    parts.append("</section>")

    # 2 — durations
    parts.append('<section id="durations">')
    parts.append("<h2>۲. زمان اجرا</h2>")
    parts.append('<table class="table">')
    parts.append(
        _render_metric_row(
            "مدت آموزش (مدل‌سازی)",
            _fmt_duration(data.training_duration_seconds),
        )
    )
    parts.append(
        _render_metric_row(
            "مدت استنباط (پایه + فاین‌تیون‌شده)",
            _fmt_duration(data.inference_duration_seconds),
        )
    )
    parts.append(
        _render_metric_row("مدت داوری", _fmt_duration(data.judge_duration_seconds))
    )
    parts.append(
        _render_metric_row("مجموع زمان هسته (آموزش + استنباط + داوری)", _fmt_duration(total_core))
    )
    parts.append("</table>")
    parts.append("</section>")

    # 3 — training config
    parts.append('<section id="training-config">')
    parts.append("<h2>۳. تنظیمات آموزش</h2>")
    parts.extend(_render_training_context(data.report_context))
    parts.append("</section>")

    # 4 — token usage + judge cost
    parts.append('<section id="tokens-cost">')
    parts.append("<h2>۴. مصرف توکن و هزینهٔ تخمینی داوری</h2>")
    parts.extend(_render_token_table(usage))
    parts.append(token_warning)
    parts.append('<table class="table cost">')
    parts.append(_render_metric_row("مدل داور", _ltr(cost.model_display or MISSING)))
    parts.append(_render_metric_row("هزینهٔ تخمینی داوری", _fmt_cost(cost)))
    if cost.kind == "range" and cost.range_label:
        parts.append(
            _render_metric_row("بازهٔ قیمت", _ltr(f"{cost.range_label} — {cost.currency} / میلیون توکن"))
        )
    parts.append("</table>")
    parts.append(
        '<p class="pricing-date">مبنای قیمت‌گذاری: '
        f'{_ltr(cost.snapshot_date) if cost.snapshot_date else _missing_or(None)}</p>'
    )
    parts.append(cost_note)
    parts.append("</section>")

    # 5 — criteria
    parts.append('<section id="criteria">')
    parts.append("<h2>۵. معیارهای ارزیابی</h2>")
    parts.extend(_render_criteria(report))
    parts.append("</section>")

    # 6 — selected samples
    parts.append('<section id="samples">')
    parts.append("<h2>۶. چند نمونهٔ منتخب</h2>")
    sample_groups = (
        ("پیروزی‌های قوی مدل فاین‌تیون‌شده", fine_wins),
        ("پیروزی‌های مدل پایه", base_wins),
    )
    rendered_any = False
    for heading, group in sample_groups:
        if not group:
            continue
        rendered_any = True
        parts.append(f"<h3>{_esc(heading)}</h3>")
        parts.extend(_render_samples(group))
    if tie is not None:
        rendered_any = True
        parts.append("<h3>مساوی</h3>")
        parts.extend(_render_samples([tie]))
    if not rendered_any:
        parts.append(f'<p class="conclusion">{_esc("نمونهٔ ارزیابی‌شده‌ای در دسترس نیست.")}</p>')
    parts.append("</section>")

    # 7 — validation & reproducibility
    parts.append('<section id="validation">')
    parts.append("<h2>۷. اعتبارسنجی و بازتولیدپذیری</h2>")
    parts.append('<table class="table">')
    parts.append(_render_metric_row("وضعیت اعتبارسنجی", _ltr(data.validation_status)))
    parts.append(_render_metric_row("بذر تصادفی (random seed)", _ltr(data.random_seed)))
    parts.append(_render_metric_row("نسخهٔ خط لوله", _ltr(data.pipeline_version)))
    parts.append(_render_metric_row("شناسهٔ اجرا (run ID)", _ltr(data.run_id)))
    parts.append("</table>")
    parts.append("</section>")

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


_CSS = """
:root {
  --bg: #f5f6f8;
  --card: #ffffff;
  --text: #1a1d21;
  --muted: #5b636e;
  --border: #e2e5ea;
  --accent: #2a4b7c;
  --missing: #9a3412;
  --incomplete: #92400e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Tahoma, "Noto Sans Arabic", "DejaVu Sans", Arial, sans-serif;
  line-height: 1.8;
  font-size: 15px;
}
main {
  max-width: 860px;
  margin: 0 auto;
  padding: 24px 16px 48px;
}
h1 { font-size: 24px; color: var(--accent); margin: 0 0 8px; }
h2 {
  font-size: 18px;
  color: var(--accent);
  border-bottom: 2px solid var(--border);
  padding-bottom: 6px;
  margin-top: 32px;
}
h3 { font-size: 16px; margin: 18px 0 6px; }
section { background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px; margin-top: 16px; }
table.table { width: 100%; border-collapse: collapse; margin: 8px 0; }
table.table th, table.table td {
  text-align: right;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}
table.table th { color: var(--muted); font-weight: normal; width: 42%; }
table.table td { font-weight: 600; }
.ltr { direction: ltr; unicode-bidi: isolate; font-variant-numeric: tabular-nums; }
.missing { color: var(--missing); font-style: italic; }
.incomplete { color: var(--incomplete); font-style: italic; }
.conclusion {
  background: #eef3fb;
  border-right: 4px solid var(--accent);
  padding: 10px 14px;
  border-radius: 6px;
  margin: 10px 0 0;
}
.note {
  color: var(--muted);
  font-size: 13px;
  margin: 10px 0 0;
}
details {
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 8px 0;
  padding: 0;
}
details summary {
  cursor: pointer;
  padding: 10px 12px;
  font-weight: 600;
  background: #fafbfc;
  border-radius: 8px;
}
.sample-body { padding: 4px 14px 14px; }
.sample-body p { margin: 10px 0; }
.sample-body .answer { display: block; white-space: pre-wrap; word-break: break-word;
  background: #f7f8fa; border-radius: 6px; padding: 8px 10px; margin-top: 4px; }
"""
