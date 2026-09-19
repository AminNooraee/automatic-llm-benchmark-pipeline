# Benchmark inference engine

The inference engine consumes only the normalized dataset contract produced by
the dataset layer:

```json
{"id":"sample-1","prompt":"Explain AI"}
```

For every sample, the engine sends one user message containing the exact same
prompt to the configured `models.base` and `models.fine_tuned` endpoints. It
uses the provider-independent `ModelClient`; the engine has no provider or
HTTP implementation logic.

## Execution and concurrency

`runtime.concurrency` limits the number of samples in flight. Its default is
`1`, which processes dataset records and their base/fine-tuned calls
sequentially. Values above `1` allow independent samples to run concurrently;
the base request still precedes the fine-tuned request within each sample.

Model request retries remain the responsibility of the existing client and are
configured with `runtime.max_retries` and `runtime.retry_backoff_seconds`.

## Artifacts and failures

The canonical output is:

```text
runs/<run_id>/
└── inference/
    ├── checkpoints.jsonl
    └── responses.jsonl
```

Each response contains `id`, `prompt`, `base_response`,
`fine_tuned_response`, and `metadata`. Metadata records the aggregate sample
status plus per-model status, model name, latency, attempts, finish reason,
token usage, and normalized error information.

A model failure does not abort the dataset run. The failed response is empty,
its error is stored in metadata, and the sample is marked `partial` or `failed`.
Checkpoint records are flushed after each completed sample. The final response
file is always written in normalized dataset order.

## Resume

Resume an existing run with the same resolved configuration:

```text
llm-benchmark run --config configs/example.yaml --resume-run runs/<run_id>
```

Fully successful samples are skipped. For a partial sample, only the failed
model role is called again; the successful response is reused. Resume rejects
unknown sample IDs, changed prompts, malformed artifacts, and configuration
fingerprint mismatches instead of combining incompatible run data.

Judge evaluation, metrics, and reporting remain separate components described
in `docs/judge-evaluation.md` and `docs/reporting.md`.
