# Configuration guide

Configuration is strict YAML. Unknown keys and invalid value types are rejected.
Relative paths are resolved from the configuration file's directory.

## Complete example

```yaml
version: 1

dataset:
  path: ../data/benchmark.jsonl
  format: auto
  # Optional custom schema mapping:
  # columns:
  #   id: record_id
  #   prompt: question_text

models:
  base:
    base_url: https://gateway.example/v1
    name: base-model
    api_key: ${BASE_API_KEY}
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 512

  fine_tuned:
    base_url: https://gateway.example/v1
    name: fine-tuned-model
    api_key: ${FINE_TUNED_API_KEY}
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 512

  judge:
    base_url: https://judge.example/v1
    name: judge-model
    api_key: ${JUDGE_API_KEY}
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 1024

evaluation:
  prompt_template: ../prompts/judge/default_pairwise.txt
  criteria:
    - name: correctness
      description: The answer is factually and logically correct.
      weight: 0.6
      minimum: 1
      maximum: 5
    - name: clarity
      description: The answer is understandable and well structured.
      weight: 0.4
      minimum: 1
      maximum: 5

runtime:
  concurrency: 1
  max_retries: 2
  retry_backoff_seconds: 0.5
  random_seed: 42
  log_level: INFO

output:
  runs_dir: ../runs
```

## Dataset

`dataset.path` must exist and use `.json`, `.jsonl`, `.csv`, or `.parquet`.
With `format: auto`, the extension selects the file adapter and record fields
are inspected to select a schema adapter. An explicit format must match the
file extension.

`dataset.columns` overrides schema detection. Dotted paths can address nested
JSON fields. See [dataset-format.md](dataset-format.md) for every supported
schema.

## Model endpoints

The three roles use the same schema. `base_url` should normally include the
gateway's API prefix, such as `/v1`; the client appends `/chat/completions`.
Only HTTP(S) OpenAI-compatible endpoints are accepted.

`api_key` is optional. Empty strings become `null`. Prefer an environment
reference such as `${BASE_API_KEY}`. A referenced variable must exist when the
configuration is loaded.

`timeout_seconds` must be greater than zero and no more than 3600. Generation
parameters are copied into the request body, but `model`, `messages`, and
`stream` are reserved. Multiple completions (`n` other than `1`) are rejected.

## Evaluation

The prompt template must be UTF-8 text with a `.txt` extension and contain:

- `{{question}}`
- `{{answer_a}}`
- `{{answer_b}}`
- `{{criteria}}`
- `{{output_schema}}`

Criterion names use lowercase letters, digits, and underscores. Names must be
unique, ranges must be increasing, every weight must be positive, and weights
must sum to `1.0` within floating-point tolerance.

## Runtime

- `concurrency`: samples in flight, from 1 to 256. The default is sequential.
- `max_retries`: retries after the initial request, from 0 to 20.
- `retry_backoff_seconds`: initial exponential backoff, from 0 to 300 seconds.
- `random_seed`: reproducible judge answer ordering.
- `log_level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.

Endpoint rate limits should determine concurrency and retry values.

## Output

`output.runs_dir` must be a writable directory if it already exists. Each run
uses a unique child directory and never overwrites another run. Resume requires
the current configuration fingerprint to match the original run.

## Preflight validation

Run:

```bash
llm-benchmark validate --config configs/example.yaml
```

Validation checks configuration shape, paths, dataset format, prompt template,
criterion math, URL structure, runtime bounds, and output-directory safety. It
does not contact model endpoints or create a run.
