# Metrics and benchmark reports

The reporting component joins `evaluation/judge_results.jsonl` with
`inference/responses.jsonl`. The join is strict: duplicate IDs, missing IDs, or
prompt mismatches stop report generation rather than producing a misleading
report.

## Model mapping

Successful judge decisions are stored using anonymous labels `A` and `B`. The
metrics calculator uses each result's `metadata.answer_order` to map:

- the winner back to `base`, `fine_tuned`, or `tie`;
- the overall A/B scores back to base and fine-tuned scores;
- every criterion's A/B scores back to base and fine-tuned scores.

The original judge label and answer order remain in each JSON sample result for
auditability.

## Aggregate metrics

Reports calculate:

- total samples;
- successful, failed, and skipped evaluations;
- base wins, fine-tuned wins, and ties;
- base, fine-tuned, and tie rates as percentages;
- average base and fine-tuned scores.

Win and tie rates use successful evaluations as the denominator. Failed judge
calls and invalid judge output count as failed evaluations. Samples without
both inference responses count as skipped evaluations. Scores are averaged
only across successful evaluations; when there are none, average scores are
`null`.

## Generated artifacts

Every completed run contains:

```text
runs/<run_id>/reports/
├── report.json
├── report.md
└── samples.csv
```

`report.json` is the complete machine-readable report. It contains benchmark
metadata, sanitized model endpoint information, dataset provenance, evaluation
criteria, aggregate metrics, and resolved per-sample results.

`report.md` is a human-readable summary with model and dataset information,
overall results, criterion definitions, and up to five sample examples.

`samples.csv` contains the sample ID, prompt, base and fine-tuned responses,
evaluation status, resolved winner, model scores, criterion scores, judge
reason, and error message. Text beginning with spreadsheet formula characters
is prefixed with an apostrophe in CSV only; JSON preserves the original text.

Charts, advanced statistical analysis, UI, databases, and distributed
execution are not included in v1.0.0. A CPU-only container definition is
provided for running the same command-line application against external model
endpoints.
