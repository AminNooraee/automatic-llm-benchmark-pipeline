# Benchmark workflow

## 1. Normalize the dataset

The selected file adapter reads JSON, JSONL, CSV, or Parquet. Schema adapters
recognize simple prompts, question/answer records, instruction/input records,
OpenAI messages, ChatML, ShareGPT, or configured custom columns. Every valid
sample becomes `{id, prompt}` and is persisted before model requests begin.

## 2. Run paired inference

The exact same prompt is sent to the base and fine-tuned endpoints. Requests
use the shared OpenAI-compatible client and each endpoint's model name, key,
timeout, and generation parameters. Responses and per-role failure metadata are
checkpointed.

## 3. Build a blind judge request

For each complete response pair, a seeded randomizer assigns the base and
fine-tuned responses to candidate labels A and B. The judge prompt contains the
original question, anonymous candidates, criteria, and required JSON schema. It
never contains the private label-to-model mapping.

## 4. Validate the judge decision

The judge response must be one JSON object with winner, A/B scores, criterion
scores, and reason. Malformed output and API failures are recorded per sample.
Samples missing one inference response are skipped rather than judged unfairly.

## 5. Resolve and aggregate

The stored answer order maps winner and scores back to `base` and
`fine_tuned`. Metrics count wins, ties, failures, and skips; calculate rates
over successful evaluations; and average model scores over successful samples.

## 6. Generate reports

One typed report model renders JSON, Markdown, and CSV. All formats originate
from the same resolved per-sample data, preventing inconsistent calculations
between reports.

## 7. Validate and seal the run

Before status becomes `completed`, the pipeline verifies required artifacts,
schemas, sample IDs, ordering, counts, report metadata, Markdown structure, CSV
row count, and secret redaction. It then records environment information and
SHA-256 digests for reproducibility.

The output is a self-contained benchmark run directory. Project #1 can change
how the fine-tuned endpoint is produced without changing this workflow.
