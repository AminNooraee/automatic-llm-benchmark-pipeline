# Automatic LLM Benchmark Pipeline

Automatic LLM Benchmark Pipeline is a reproducible command-line system for
measuring whether a fine-tuned language model improves on its base model. It
sends the same benchmark prompts to both models, asks an independent judge
model to compare anonymous answers, resolves the decisions back to the real
model roles, and produces audit-ready JSON, Markdown, and CSV reports.

The pipeline consumes model endpoints; it does not train or serve models.
Project #1 can produce the fine-tuned model, while this independent Project #2
compares that endpoint with its base model.

The runtime knows endpoint URLs and model routing names, not model families or
providers. Base, fine-tuned, and judge are the same `EndpointConfig` concept and
must expose non-streaming OpenAI-compatible Chat Completions.

## One-command benchmark

After preparing a repository-local config/dataset and exporting any `${VAR}`
secrets referenced by the config, run:

```sh
sh scripts/run_benchmark.sh configs/company_benchmark.yaml
```

The launcher builds the CPU-only image when absent, runs an ephemeral non-root
container, performs endpoint preflight, and writes JSON, Markdown, and CSV under
`runs/`. It does not mount the Docker socket or require a GPU or VM. An optional
Project #1 handoff is accepted with `--project1-handoff /path/to/gateway_manifest.json`.

Current release: **v1.0.0**

## Architecture

```text
Benchmark dataset (JSON / JSONL / CSV / Parquet)
                    |
                    v
        Source and format adapters
                    |
                    v
       Schema detection and normalization
                    |
                    v
        normalized_dataset.jsonl
                    |
          +---------+---------+
          |                   |
          v                   v
   Base model endpoint   Fine-tuned endpoint
          |                   |
          +---------+---------+
                    |
                    v
             Inference engine
                    |
                    v
              responses.jsonl
                    |
                    v
       Anonymous, randomized A/B judge
                    |
                    v
            judge_results.jsonl
                    |
                    v
       Metrics and mapping resolution
                    |
                    v
      report.json + report.md + samples.csv
```

All three model roles use the same non-streaming OpenAI-compatible Chat
Completions contract. The benchmark code does not branch on provider type.

## Key capabilities

- Normalizes JSON, JSONL, CSV, and Parquet datasets into `{id, prompt}`.
- Detects prompt, question/answer, instruction/input, OpenAI messages, ChatML,
  and ShareGPT records.
- Supports explicit custom prompt and ID column mappings, including dotted
  paths for nested records.
- Sends each prompt to configured base and fine-tuned endpoints with isolated
  failure handling and resumable checkpoints.
- Randomizes answer order before judge evaluation so model identity is hidden.
- Validates the judge's structured JSON response and maps A/B results back to
  base and fine-tuned roles.
- Generates aggregate metrics, per-sample results, integrity validation, and
  reproducibility metadata.
- Redacts configured API keys from snapshots, logs, and metadata.

## Requirements

- Python 3.11 or 3.12
- Network access from the runner to three OpenAI-compatible endpoints
- PyArrow, installed with the package, for Parquet input
- Docker is optional

No model weights or provider SDKs are required.

## Quick start

Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Copy [`configs/example.yaml`](configs/example.yaml), replace the example
endpoint details, and validate the resolved configuration:

```bash
llm-benchmark validate --config configs/example.yaml
```

Run the complete benchmark:

```bash
llm-benchmark run --config configs/example.yaml
```

The command creates one uniquely named child directory under the configured
`output.runs_dir`. The example configuration points to local placeholder
endpoints, so those servers must be running or replaced before execution.

To continue a run that contains failed inference or judge samples:

```bash
llm-benchmark run --config configs/example.yaml --resume-run runs/<run_id>
```

Resume requires the same resolved configuration fingerprint. Successful work
is reused; failed work is retried.

## Configuration

Configuration is strict YAML. Unknown keys, invalid paths, incompatible
criteria, unsafe generation settings, and malformed endpoint URLs fail during
preflight validation. Relative paths are resolved from the YAML file's
directory.

A root-level `benchmark.yaml` can use:

```yaml
version: 1

dataset:
  path: examples/datasets/benchmark.jsonl
  format: auto
  # Optional:
  # columns:
  #   id: record_id
  #   prompt: question_text

models:
  base:
    base_url: http://localhost:8001/v1
    name: base-model
    api_key: null
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 512

  fine_tuned:
    base_url: http://localhost:8002/v1
    name: fine-tuned-model
    api_key: null
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 512

  judge:
    base_url: http://localhost:8003/v1
    name: judge-model
    api_key: null
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 1024

evaluation:
  prompt_template: prompts/judge/default_pairwise.txt
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
  runs_dir: runs
```

For secrets, set `api_key` to an environment-variable reference instead of
committing a key. See the [configuration guide](docs/configuration.md) for the
complete schema, environment expansion, constraints, and path-resolution
rules.

## OpenAI-compatible model connections

Every model role is configured with only:

- `base_url`, normally including the `/v1` prefix;
- `name`, sent as the request's model identifier;
- optional `api_key`;
- `timeout_seconds`;
- `generation_parameters`.

The shared client sends `POST <base_url>/chat/completions` and reads the
standard assistant content from the response. Therefore LiteLLM proxies, vLLM
OpenAI-compatible servers, the OpenAI API, and internal company gateways can
serve any role without provider-specific pipeline code.

| Logical role | Purpose |
| --- | --- |
| `models.base` | Generates the baseline answer |
| `models.fine_tuned` | Generates the candidate fine-tuned answer |
| `models.judge` | Evaluates the two anonymous candidate answers |

Compatibility depends on the endpoint implementing the OpenAI-compatible Chat
Completions request and response contract used by the client.

## Supported datasets

### File formats

| Format | Accepted structure |
| --- | --- |
| JSON | One object, an array of objects, or a top-level message sequence |
| JSONL | One object per non-empty line |
| CSV | Header row followed by records |
| Parquet | Tabular records read with PyArrow |

### Record schemas

- Simple prompt: `{"prompt": "Explain AI"}`
- Question/answer: `question` becomes the prompt; the reference `answer` is not
  sent to the compared models
- Instruction/input: non-empty `instruction` and `input` are combined
- OpenAI messages: the last user message is selected
- ChatML: structured or serialized ChatML is recognized
- ShareGPT: the last human/user message is selected
- Custom mapping: `dataset.columns.prompt` and optional
  `dataset.columns.id` override schema detection

Ambiguous schemas, malformed records, empty prompts, duplicate IDs, unsupported
formats, and missing mapped fields are rejected with source and row context.

## Run artifacts

```text
runs/<run_id>/
|-- config.snapshot.json
|-- manifest.json
|-- dataset/
|   |-- original_dataset.*
|   +-- normalized_dataset.jsonl
|-- inference/
|   |-- checkpoints.jsonl
|   +-- responses.jsonl
|-- evaluation/
|   +-- judge_results.jsonl
|-- reports/
|   |-- report.json
|   |-- report.md
|   +-- samples.csv
|-- metadata/
|   |-- environment.json
|   |-- artifact_validation.json
|   +-- reproducibility.json
+-- logs/
    +-- run.jsonl
```

The sanitized configuration snapshot and manifest describe what ran.
Environment metadata records the Python, operating system, CPU, and direct
dependency versions. Reproducibility metadata records configuration, dataset,
prompt, and artifact hashes without storing configured secrets.

## Understanding reports

- `reports/report.md` is the reviewer-friendly benchmark summary, including
  compared models, dataset provenance, overall metrics, criteria, and examples.
- `reports/report.json` is the complete machine-readable report, including
  benchmark metadata, aggregate metrics, and resolved per-sample results.
- `reports/samples.csv` supports spreadsheet or notebook review of each prompt,
  both responses, resolved winner, scores, reason, and error status.

Version 1.0.0 reports total, successful, failed, and skipped evaluations; base
wins, fine-tuned wins, and ties; win/tie rates over successful evaluations;
and average model scores. It does not calculate confidence intervals or
statistical significance.

## Reproducibility and failure handling

- `runtime.random_seed` makes judge answer ordering repeatable.
- Inference responses are checkpointed after each completed sample.
- Transient request errors use bounded retries and exponential backoff.
- One failed model or judge request does not discard other completed samples.
- Resume validates configuration and artifact compatibility before reuse.
- Completion requires cross-artifact IDs, ordering, counts, schemas, report
  structure, and secret-redaction checks to pass.

## Docker

Build the CPU-only runner:

```bash
docker build -t automatic-llm-benchmark-pipeline:1.0.0 .
```

The image contains the benchmark application, not model weights or a model
server. Mount configuration, datasets, prompts, and an output directory; the
container calls external OpenAI-compatible endpoints. See the
[Docker guide](docs/docker.md) for full commands and host-network guidance.

## Testing and acceptance

```bash
python -m pytest
```

The v1.0.0 acceptance baseline is 93 passing tests across unit, integration,
CLI, release, and complete mocked HTTP end-to-end coverage. The end-to-end test
verifies dataset normalization, all three model roles, metrics, reports,
artifact integrity, secret redaction, and reproducibility metadata.

See the [Project #2 acceptance report](PROJECT_2_ACCEPTANCE_REPORT.md) and
[v1.0.0 release notes](RELEASE_NOTES.md) for handover and release status.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration guide](docs/configuration.md)
- [Usage examples](docs/usage.md)
- [Benchmark workflow](docs/benchmark-workflow.md)
- [Dataset formats](docs/dataset-format.md)
- [Model client](docs/model-client.md)
- [Inference engine](docs/inference-engine.md)
- [Judge evaluation](docs/judge-evaluation.md)
- [Metrics and reporting](docs/reporting.md)
- [Docker deployment](docs/docker.md)
- [Release checklist](RELEASE_CHECKLIST.md)
- [Changelog](CHANGELOG.md)

## Scope and known limitations

The v1.0.0 core is intentionally a finite local command-line workflow. It does
not include a UI, database, distributed executor, human-evaluation workflow,
plugin framework, bundled model serving, cloud deployment, charts, or advanced
statistical analysis. Production endpoint compatibility and credentials must
be validated in the target environment. The Docker definition is provided,
but the image was not built during local acceptance because Docker was
unavailable.

`llm-benchmark run` checks base, fine-tuned, and judge compatibility before it
creates a run. Use `llm-benchmark preflight --config CONFIG` for the check alone,
or the explicit `--skip-preflight` run override. A reasoning-capable endpoint may
return extra reasoning metadata, but must still provide final assistant
`content`; private reasoning is never substituted or persisted. Operator-known
request options remain available through `generation_parameters`.

Preflight verifies connectivity and response-contract compatibility only; it
does not establish model quality, capacity, or production readiness. Real
endpoint/network behavior must be validated in the operator's environment.
