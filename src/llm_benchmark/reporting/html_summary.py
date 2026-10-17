"""Shared bilingual, self-contained benchmark dashboard renderer."""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from llm_benchmark.config.handoff import ReportContext
from llm_benchmark.metrics.models import ResolvedSampleResult
from llm_benchmark.reporting.cost import JudgeCost, JudgeCostComparison
from llm_benchmark.reporting.models import BenchmarkReport
from llm_benchmark.reporting.token_usage import RoleUsage, TokenUsageSummary

Locale = Literal["fa", "en"]

_LABELS = {
    "fa": {
        "dir": "rtl", "title": "خلاصه بنچمارک مدل",
        "subtitle": "نمای مدیریتی مقایسه مدل پایه و مدل فاین‌تیون‌شده",
        "summary": "خلاصهٔ نتیجه", "durations": "زمان اجرا",
        "training": "تنظیمات آموزش",
        "tokens_cost": "مصرف توکن و هزینهٔ تخمینی داوری",
        "criteria": "معیارهای ارزیابی", "samples": "چند نمونهٔ منتخب",
        "validation": "اعتبارسنجی و بازتولیدپذیری",
        "total_samples": "مجموع نمونه‌ها", "fine_wins": "پیروزی مدل فاین‌تیون‌شده",
        "base_wins": "پیروزی مدل پایه", "score_lift": "بهبود امتیاز",
        "fine": "فاین‌تیون‌شده", "base": "مدل پایه", "tie": "مساوی",
        "score_comparison": "مقایسه میانگین امتیاز",
        "training_duration": "مدت آموزش", "inference_duration": "مدت استنباط",
        "judge_duration": "مدت داوری", "total_duration": "مجموع زمان هسته",
        "minute": "دقیقه", "second": "ثانیه", "model_name": "نام مدل پایه",
        "fine_model": "مدل فاین‌تیون‌شده", "epochs": "تعداد اپاک",
        "learning_rate": "نرخ یادگیری", "batch_size": "اندازه دسته",
        "cutoff_len": "طول برش", "lora_rank": "رتبه LoRA",
        "lora_alpha": "ضریب LoRA", "lora_dropout": "ریزش LoRA",
        "role": "نقش", "input_tokens": "توکن ورودی",
        "output_tokens": "توکن خروجی", "total_tokens": "مجموع توکن",
        "coverage": "پوشش", "judge": "داور", "actual_judge": "مدل داور واقعی",
        "same_workload": "برآورد هزینه همین حجم داوری",
        "pricing_basis": "مبنای قیمت‌گذاری",
        "cost_disclaimer": "هزینه‌ها بر پایه مصرف توکن گزارش‌شده داور تخمین زده شده‌اند و روش توکن‌سازی ارائه‌دهندگان ممکن است متفاوت باشد.",
        "question": "سؤال", "base_answer": "پاسخ مدل پایه",
        "fine_answer": "پاسخ Fine-tuned", "winner": "برنده",
        "scores": "امتیازها", "judge_reason": "دلیل Judge",
        "fine_samples": "پیروزی‌های قوی مدل فاین‌تیون‌شده",
        "base_samples": "پیروزی‌های مدل پایه", "tie_samples": "نمونه مساوی",
        "no_samples": "نمونه ارزیابی‌شده‌ای در دسترس نیست.",
        "validation_status": "وضعیت اعتبارسنجی", "run_id": "شناسه اجرا",
        "pipeline_version": "نسخه خط لوله", "random_seed": "بذر تصادفی",
        "technical_details": "جزئیات فنی", "generated_at": "زمان تولید",
        "config_fingerprint": "اثر انگشت پیکربندی", "dataset_hash": "هش مجموعه داده",
        "dataset_path": "مسیر مجموعه داده", "missing": "ناموجود",
        "incomplete": "ناقص", "pass": "موفق",
    },
    "en": {
        "dir": "ltr", "title": "Model Benchmark Summary",
        "subtitle": "Executive comparison of the base and fine-tuned models",
        "summary": "Executive Summary", "durations": "Execution Times",
        "training": "Training Configuration",
        "tokens_cost": "Token Usage and Estimated Judge Cost",
        "criteria": "Evaluation Criteria", "samples": "Representative Samples",
        "validation": "Validation and Reproducibility",
        "total_samples": "Total Samples", "fine_wins": "Fine-tuned Wins",
        "base_wins": "Base Wins", "score_lift": "Score Lift",
        "fine": "Fine-tuned", "base": "Base", "tie": "Tie",
        "score_comparison": "Average Score Comparison",
        "training_duration": "Training Duration", "inference_duration": "Inference Duration",
        "judge_duration": "Judge Duration", "total_duration": "Total Core Duration",
        "minute": "min", "second": "sec", "model_name": "Base Model",
        "fine_model": "Fine-tuned Model", "epochs": "Epochs",
        "learning_rate": "Learning Rate", "batch_size": "Batch Size",
        "cutoff_len": "Cutoff Length", "lora_rank": "LoRA Rank",
        "lora_alpha": "LoRA Alpha", "lora_dropout": "LoRA Dropout",
        "role": "Role", "input_tokens": "Input Tokens",
        "output_tokens": "Output Tokens", "total_tokens": "Total Tokens",
        "coverage": "Coverage", "judge": "Judge", "actual_judge": "Actual Judge Model",
        "same_workload": "Estimated cost for this judge workload",
        "pricing_basis": "Pricing snapshot",
        "cost_disclaimer": "Costs are estimated from the tokens reported by the current judge; provider tokenization may differ.",
        "question": "Question", "base_answer": "Base Model Answer",
        "fine_answer": "Fine-tuned Answer", "winner": "Winner",
        "scores": "Scores", "judge_reason": "Judge Reason",
        "fine_samples": "Strong Fine-tuned Wins", "base_samples": "Base Wins",
        "tie_samples": "Tie", "no_samples": "No evaluated samples are available.",
        "validation_status": "Validation Status", "run_id": "Run ID",
        "pipeline_version": "Pipeline Version", "random_seed": "Random Seed",
        "technical_details": "Technical Details", "generated_at": "Generated At",
        "config_fingerprint": "Configuration Fingerprint", "dataset_hash": "Dataset Hash",
        "dataset_path": "Dataset Path", "missing": "Unavailable",
        "incomplete": "Incomplete", "pass": "PASS",
    },
}


@dataclass(frozen=True)
class SummaryData:
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
    actual_judge_name: str
    judge_costs: JudgeCostComparison


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _ltr(value: object) -> str:
    return f'<span class="ltr" dir="ltr">{_esc(value)}</span>'


def _auto(value: object, css_class: str = "user-content") -> str:
    return f'<span class="{css_class}" dir="auto">{_esc(value)}</span>'


def _missing(labels: dict[str, str]) -> str:
    return f'<span class="missing">{labels["missing"]}</span>'


def _number(value: float | None, labels: dict[str, str], digits: int = 4) -> str:
    return _missing(labels) if value is None else _ltr(f"{value:.{digits}g}")


def _duration(value: float | None, labels: dict[str, str]) -> str:
    if value is None:
        return _missing(labels)
    minutes, seconds = divmod(round(max(0.0, float(value))), 60)
    if minutes:
        return f'{_ltr(minutes)} {labels["minute"]} · {_ltr(seconds)} {labels["second"]}'
    return f'{_ltr(seconds)} {labels["second"]}'


def _token(usage: RoleUsage, field: str, labels: dict[str, str]) -> str:
    if usage.missing:
        return _missing(labels)
    if not usage.complete:
        return f'<span class="incomplete">{labels["incomplete"]}</span>'
    value = getattr(usage, field)
    return _missing(labels) if value is None else _ltr(f"{value:,}")


def _cost(value: JudgeCost, labels: dict[str, str]) -> str:
    if not value.priced:
        return _missing(labels)
    if value.kind == "flat" and value.cost is not None:
        return _ltr(f"{value.cost:.6f} {value.currency}")
    if value.kind == "range" and value.cost_low is not None and value.cost_high is not None:
        return _ltr(f"{value.cost_low:.6f} – {value.cost_high:.6f} {value.currency}")
    return _missing(labels)


def _winner(value: str | None, labels: dict[str, str]) -> str:
    if value is None:
        return labels["missing"]
    return {"base": labels["base"], "fine_tuned": labels["fine"], "tie": labels["tie"]}.get(value, value)


def _conclusion(report: BenchmarkReport, locale: Locale) -> str:
    metrics = report.aggregate_metrics
    if metrics.successful_evaluations == 0:
        return "هیچ ارزیابی موفقی برای نتیجه‌گیری وجود ندارد." if locale == "fa" else "No successful evaluations are available."
    if metrics.fine_tuned_model_wins > metrics.base_model_wins:
        return "مدل فاین‌تیون‌شده عملکرد بهتری نشان داد." if locale == "fa" else "The fine-tuned model performed better."
    if metrics.base_model_wins > metrics.fine_tuned_model_wins:
        return "مدل پایه عملکرد بهتری نشان داد." if locale == "fa" else "The base model performed better."
    return "عملکرد دو مدل هم‌ارز بود." if locale == "fa" else "The two models performed equally."


def _select_samples(samples: list[ResolvedSampleResult]):
    successful = [s for s in samples if s.evaluation_status.value == "success" and s.winner is not None]

    def gap(sample: ResolvedSampleResult, fine_wins: bool) -> float:
        difference = (sample.fine_tuned_score or 0) - (sample.base_score or 0)
        return difference if fine_wins else -difference

    fine = sorted((s for s in successful if s.winner.value == "fine_tuned"), key=lambda s: gap(s, True), reverse=True)[:3]
    base = sorted((s for s in successful if s.winner.value == "base"), key=lambda s: gap(s, False), reverse=True)[:2]
    tie = [s for s in successful if s.winner.value == "tie"][:1]
    return fine, base, tie


def _criterion_averages(report: BenchmarkReport, name: str):
    pairs = [(s.criteria_scores or {}).get(name) for s in report.per_sample_results]
    scores = [value for value in pairs if value is not None]
    if not scores:
        return None, None
    return (
        sum(value.base_model for value in scores) / len(scores),
        sum(value.fine_tuned_model for value in scores) / len(scores),
    )


def _render_training(data: SummaryData, labels: dict[str, str]) -> str:
    training = data.report_context.training if data.report_context else None
    lora = training.lora if training else None
    rows = (
        (labels["model_name"], _ltr(training.model_name) if training else _missing(labels)),
        (labels["fine_model"], _ltr(data.report.model_information.fine_tuned.name)),
        (labels["epochs"], _ltr(training.epochs) if training else _missing(labels)),
        (labels["learning_rate"], _ltr(training.learning_rate) if training else _missing(labels)),
        (labels["batch_size"], _ltr(training.batch_size) if training else _missing(labels)),
        (labels["cutoff_len"], _ltr(training.cutoff_len) if training else _missing(labels)),
        (labels["lora_rank"], _ltr(lora.rank) if lora else _missing(labels)),
        (labels["lora_alpha"], _ltr(lora.alpha) if lora else _missing(labels)),
        (labels["lora_dropout"], _ltr(lora.dropout) if lora else _missing(labels)),
    )
    return "".join(f'<div class="meta-item"><span>{label}</span><strong>{value}</strong></div>' for label, value in rows)


def _render_tokens(data: SummaryData, labels: dict[str, str]) -> str:
    rows = ((labels["base"], data.token_summary.base), (labels["fine"], data.token_summary.fine_tuned), (labels["judge"], data.token_summary.judge))
    body = []
    for name, usage in rows:
        coverage = _missing(labels) if usage.missing else (_ltr(f"{usage.available}/{usage.expected_successful}") if usage.complete else f'<span class="incomplete">{labels["incomplete"]}</span>')
        body.append(f"<tr><td>{name}</td><td>{_token(usage, 'prompt_tokens', labels)}</td><td>{_token(usage, 'completion_tokens', labels)}</td><td>{_token(usage, 'total_tokens', labels)}</td><td>{coverage}</td></tr>")
    return f'<div class="table-wrap"><table><thead><tr><th>{labels["role"]}</th><th>{labels["input_tokens"]}</th><th>{labels["output_tokens"]}</th><th>{labels["total_tokens"]}</th><th>{labels["coverage"]}</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _render_costs(data: SummaryData, labels: dict[str, str]) -> str:
    rows = "".join(f'<div class="price-row"><span class="ltr" dir="ltr">{_esc(item.model_display)}</span><strong>{_cost(item, labels)}</strong></div>' for item in data.judge_costs.estimates)
    if not rows:
        rows = f'<div class="price-row">{_missing(labels)}</div>'
    snapshot = _ltr(data.judge_costs.snapshot_date) if data.judge_costs.snapshot_date else _missing(labels)
    return f'<div class="judge-card"><span>{labels["actual_judge"]}</span><strong>{_ltr(data.actual_judge_name)}</strong></div><div class="pricing"><h3>{labels["same_workload"]}</h3><div class="price-list">{rows}</div><p class="pricing-date">{labels["pricing_basis"]}: {snapshot}</p><p class="note">{labels["cost_disclaimer"]}</p></div>'


def _render_criteria(data: SummaryData, labels: dict[str, str]) -> str:
    cards = []
    for criterion in data.report.evaluation_criteria:
        base, fine = _criterion_averages(data.report, criterion.name)
        cards.append(f'<article class="criterion-card"><div class="criterion-head"><strong>{_auto(criterion.name)}</strong><span>{_ltr(f"{criterion.weight * 100:.0f}%")}</span></div><p>{_auto(criterion.description)}</p><div class="criterion-scores"><span>{labels["base"]}<b>{_number(base, labels)}</b></span><span>{labels["fine"]}<b>{_number(fine, labels)}</b></span></div></article>')
    return "".join(cards)


def _render_sample(sample: ResolvedSampleResult, labels: dict[str, str]) -> str:
    winner = _winner(sample.winner.value if sample.winner else None, labels)
    base_score, fine_score = _number(sample.base_score, labels), _number(sample.fine_tuned_score, labels)
    reason = sample.judge_reason or sample.error_message or labels["missing"]
    return f'<details class="sample-card"><summary><span>{_ltr(sample.id)}</span><span class="winner">{labels["winner"]}: {winner}</span><span>{labels["scores"]}: {base_score} / {fine_score}</span></summary><div class="sample-body"><div><strong>{labels["question"]}</strong>{_auto(sample.prompt)}</div><div><strong>{labels["base_answer"]}</strong>{_auto(sample.base_response, "answer")}</div><div><strong>{labels["fine_answer"]}</strong>{_auto(sample.fine_tuned_response, "answer")}</div><div><strong>{labels["judge_reason"]}</strong>{_auto(reason)}</div></div></details>'


def render_benchmark_summary_html(data: SummaryData, *, locale: Locale = "fa") -> str:
    """Render either locale from one data model and one layout implementation."""

    labels, report = _LABELS[locale], data.report
    metrics, averages, rates = report.aggregate_metrics, report.aggregate_metrics.average_scores_per_model, report.aggregate_metrics.win_rate_percentage
    lift = averages.fine_tuned_model - averages.base_model if averages.base_model is not None and averages.fine_tuned_model is not None else None
    total_duration = data.training_duration_seconds + data.inference_duration_seconds + data.judge_duration_seconds if data.training_duration_seconds is not None and data.inference_duration_seconds is not None and data.judge_duration_seconds is not None else None
    fine_samples, base_samples, tie_samples = _select_samples(report.per_sample_results)
    groups = ((labels["fine_samples"], fine_samples), (labels["base_samples"], base_samples), (labels["tie_samples"], tie_samples))
    samples = "".join(f"<h3>{heading}</h3>" + "".join(_render_sample(item, labels) for item in items) for heading, items in groups if items) or f'<p class="empty">{labels["no_samples"]}</p>'
    state = "ok" if data.validation_status.lower() == "passed" else "warn"
    return f"""<!doctype html><html lang="{locale}" dir="{labels['dir']}"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{labels['title']}</title><style>{_CSS}</style></head><body><main>
<header class="hero"><div><p class="eyebrow">LLM BENCHMARK</p><h1>{labels['title']}</h1><p>{labels['subtitle']}</p></div><span class="status {state}">{labels['pass'] if state == 'ok' else _esc(data.validation_status)}</span></header>
<section id="summary"><h2>{labels['summary']}</h2><div class="kpi-grid">
<article class="kpi"><span>{labels['total_samples']}</span><strong>{_ltr(metrics.total_samples)}</strong></article>
<article class="kpi fine"><span>{labels['fine_wins']}</span><strong>{_ltr(metrics.fine_tuned_model_wins)}</strong></article>
<article class="kpi base"><span>{labels['base_wins']}</span><strong>{_ltr(metrics.base_model_wins)}</strong></article>
<article class="kpi"><span>{labels['score_lift']}</span><strong>{_number(lift, labels)}</strong></article></div>
<div class="win-bar"><span class="fine-segment" style="width:{rates.fine_tuned_model:.4f}%"></span><span class="tie-segment" style="width:{rates.ties:.4f}%"></span><span class="base-segment" style="width:{rates.base_model:.4f}%"></span></div>
<div class="legend"><span class="fine-dot">{labels['fine']}</span><span class="tie-dot">{labels['tie']}</span><span class="base-dot">{labels['base']}</span></div>
<div class="score-panel"><h3>{labels['score_comparison']}</h3><div class="score-row"><span>{labels['base']}</span><div><i style="width:{(averages.base_model or 0) / 5 * 100:.2f}%"></i></div>{_number(averages.base_model, labels)}</div><div class="score-row fine"><span>{labels['fine']}</span><div><i style="width:{(averages.fine_tuned_model or 0) / 5 * 100:.2f}%"></i></div>{_number(averages.fine_tuned_model, labels)}</div></div><p class="conclusion">{_conclusion(report, locale)}</p></section>
<section id="durations"><h2>{labels['durations']}</h2><div class="four-grid"><article><span>{labels['training_duration']}</span><strong>{_duration(data.training_duration_seconds, labels)}</strong></article><article><span>{labels['inference_duration']}</span><strong>{_duration(data.inference_duration_seconds, labels)}</strong></article><article><span>{labels['judge_duration']}</span><strong>{_duration(data.judge_duration_seconds, labels)}</strong></article><article><span>{labels['total_duration']}</span><strong>{_duration(total_duration, labels)}</strong></article></div></section>
<section id="training-config"><h2>{labels['training']}</h2><div class="meta-grid">{_render_training(data, labels)}</div></section>
<section id="tokens-cost"><h2>{labels['tokens_cost']}</h2>{_render_tokens(data, labels)}<div class="cost-panel">{_render_costs(data, labels)}</div></section>
<section id="criteria"><h2>{labels['criteria']}</h2><div class="criteria-grid">{_render_criteria(data, labels)}</div></section>
<section id="samples"><h2>{labels['samples']}</h2>{samples}</section>
<section id="validation"><h2>{labels['validation']}</h2><div class="validation-grid"><article><span>{labels['validation_status']}</span><strong class="status {state}">{labels['pass'] if state == 'ok' else _esc(data.validation_status)}</strong></article><article><span>{labels['run_id']}</span><strong>{_ltr(data.run_id)}</strong></article><article><span>{labels['pipeline_version']}</span><strong>{_ltr(data.pipeline_version)}</strong></article><article><span>{labels['random_seed']}</span><strong>{_ltr(data.random_seed)}</strong></article></div>
<details class="technical"><summary>{labels['technical_details']}</summary><div class="meta-grid"><div class="meta-item"><span>{labels['generated_at']}</span><strong>{_ltr(data.generated_at.isoformat())}</strong></div><div class="meta-item"><span>{labels['config_fingerprint']}</span><strong>{_ltr(report.benchmark_metadata.config_fingerprint)}</strong></div><div class="meta-item"><span>{labels['dataset_hash']}</span><strong>{_ltr(report.dataset_information.source_sha256)}</strong></div><div class="meta-item"><span>{labels['dataset_path']}</span><strong>{_ltr(report.dataset_information.source_path)}</strong></div></div></details></section>
</main></body></html>"""


_CSS = """
:root{color-scheme:dark;--bg:#080d18;--panel:#111827;--panel2:#172033;--line:#28344a;--text:#eef2ff;--muted:#9aa8bd;--fine:#22c55e;--base:#60a5fa;--tie:#f59e0b;--accent:#a78bfa;--danger:#fb7185}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#17213a 0,#080d18 42%);color:var(--text);font-family:Tahoma,Arial,"Noto Sans Arabic","DejaVu Sans",sans-serif;line-height:1.65}main{max-width:1180px;margin:auto;padding:32px 20px 64px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:22px}.hero h1{margin:2px 0;font-size:clamp(28px,4vw,44px)}.hero p{margin:0;color:var(--muted)}.eyebrow{color:var(--accent)!important;letter-spacing:.18em;font-size:12px}section{background:linear-gradient(145deg,rgba(23,32,51,.95),rgba(13,20,34,.96));border:1px solid var(--line);border-radius:18px;padding:22px;margin:16px 0;box-shadow:0 18px 45px rgba(0,0,0,.18)}h2{margin:0 0 18px;font-size:20px}h3{margin:18px 0 10px;font-size:15px;color:#cbd5e1}.kpi-grid,.four-grid,.validation-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi,.four-grid article,.validation-grid article,.judge-card{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:16px}.kpi span,.four-grid span,.validation-grid span,.judge-card>span,.meta-item>span{display:block;color:var(--muted);font-size:12px}.kpi strong{display:block;font-size:29px;margin-top:4px}.kpi.fine strong{color:var(--fine)}.kpi.base strong{color:var(--base)}.win-bar{display:flex;height:14px;margin:22px 0 8px;overflow:hidden;border-radius:999px;background:#29344a}.fine-segment{background:var(--fine)}.tie-segment{background:var(--tie)}.base-segment{background:var(--base)}.legend{display:flex;gap:22px;color:var(--muted);font-size:12px}.legend span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-inline-end:7px}.fine-dot:before{background:var(--fine)}.tie-dot:before{background:var(--tie)}.base-dot:before{background:var(--base)}.score-row{display:grid;grid-template-columns:130px 1fr 70px;align-items:center;gap:12px;margin:10px 0}.score-row>div{height:9px;background:#263249;border-radius:10px;overflow:hidden}.score-row i{display:block;height:100%;background:var(--base)}.score-row.fine i{background:var(--fine)}.conclusion{border-inline-start:3px solid var(--accent);background:#181d33;padding:12px 15px;border-radius:8px}.four-grid strong,.validation-grid strong{display:block;margin-top:5px}.meta-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden}.meta-item{background:var(--panel);padding:13px 15px;min-width:0}.meta-item strong{display:block;overflow-wrap:anywhere}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse}th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}th{color:var(--muted);font-size:12px}.cost-panel{display:grid;grid-template-columns:minmax(220px,.7fr) 1.3fr;gap:16px;margin-top:20px}.judge-card{display:flex;flex-direction:column;justify-content:center}.judge-card strong{font-size:19px;margin-top:8px}.price-list{display:grid;gap:8px}.price-row{display:flex;justify-content:space-between;gap:18px;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:11px 14px}.pricing-date,.note{margin:5px 0;color:var(--muted);font-size:12px}.criteria-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.criterion-card{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:15px}.criterion-head,.criterion-scores{display:flex;justify-content:space-between;gap:12px}.criterion-card p{color:var(--muted);font-size:12px;min-height:38px}.criterion-scores span{font-size:12px;color:var(--muted)}.criterion-scores b{display:block;color:var(--text);font-size:16px}.sample-card{border:1px solid var(--line);border-radius:12px;margin:9px 0;background:var(--panel)}.sample-card summary{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;cursor:pointer;padding:13px 15px}.winner{color:var(--accent)}.sample-body{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 15px 15px}.sample-body>div{background:var(--panel2);border-radius:10px;padding:12px;min-width:0}.sample-body strong{display:block;color:var(--muted);font-size:12px}.user-content,.answer{display:block;white-space:pre-wrap;overflow-wrap:anywhere}.status{display:inline-flex!important;width:max-content;padding:5px 10px;border-radius:999px;font-size:12px}.status.ok{background:#123723;color:#86efac}.status.warn{background:#422006;color:#fdba74}.technical{margin-top:14px}.technical summary{cursor:pointer;color:var(--muted)}.technical .meta-grid{margin-top:12px}.ltr{direction:ltr;unicode-bidi:isolate;font-variant-numeric:tabular-nums}.missing{color:var(--danger)}.incomplete{color:var(--tie)}.empty{color:var(--muted)}
@media(max-width:800px){main{padding:20px 12px 40px}.kpi-grid,.four-grid,.validation-grid{grid-template-columns:1fr 1fr}.meta-grid,.criteria-grid{grid-template-columns:1fr 1fr}.cost-panel{display:block}.judge-card{margin-bottom:16px}.sample-body{grid-template-columns:1fr}.score-row{grid-template-columns:90px 1fr 60px}}@media(max-width:480px){.hero{align-items:flex-start;flex-direction:column}.kpi-grid,.four-grid,.validation-grid,.meta-grid,.criteria-grid{grid-template-columns:1fr}.price-row{align-items:flex-start;flex-direction:column}.sample-card summary{flex-direction:column}}@media print{body{background:#fff;color:#111}section,.kpi,.four-grid article,.validation-grid article,.judge-card,.criterion-card,.sample-card,.meta-item,.price-row{background:#fff;color:#111;box-shadow:none}section,.sample-card{break-inside:avoid}.technical{display:none}}
"""
