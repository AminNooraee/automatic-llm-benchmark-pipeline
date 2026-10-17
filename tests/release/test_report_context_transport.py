from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_optional_report_context_is_mounted_forwarded_and_not_logged() -> None:
    launcher = (ROOT / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8")
    assert "dst=/handoff/report_context.json,readonly" in launcher
    assert "--project1-report-context /handoff/report_context.json" in launcher

    ci = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    assert "PROJECT1_REPORT_CONTEXT_B64" in ci
    assert "--project1-report-context" in ci
    assert 'run_benchmark_extra="--project1-report-context $report_context_file"' in ci
    assert 'rm -f -- "$handoff_file" "$benchmark_log" "$report_context_file"' in ci
    assert 'echo "$PROJECT1_REPORT_CONTEXT_B64"' not in ci
