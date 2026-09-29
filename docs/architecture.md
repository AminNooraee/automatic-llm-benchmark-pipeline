# Architecture

## System boundary

The pipeline consumes datasets and external OpenAI-compatible endpoints. It
does not train, serve, download, or package models. Project #1 produces the
fine-tuned endpoint; this project compares that endpoint with its base model.

```text
Dataset source
    │
    ▼
Format adapter → schema normalization → normalized_dataset.jsonl
    │
    ▼
Inference engine → base endpoint + fine-tuned endpoint → responses.jsonl
    │
    ▼
Judge engine → randomized anonymous A/B prompt → judge_results.jsonl
    │
    ▼
Metrics → resolve A/B to model roles → aggregate analysis
    │
    ▼
Reports → report.json + report.md + samples.csv
```

## Package responsibilities

- `config`: strict YAML models, environment expansion, path resolution, and
  semantic preflight validation.
- `datasets`: local source adapters, format detection, schema detection, and
  normalization into `{id, prompt}`.
- `clients`: the provider-independent asynchronous Chat Completions contract,
  HTTP implementation, retries, error normalization, and response parsing.
- `inference`: identical-prompt base/fine-tuned execution, failure isolation,
  checkpoints, concurrency limits, and resume behavior.
- `evaluation`: blind answer randomization, prompt rendering, strict judge JSON
  parsing, and evaluation artifacts.
- `metrics`: A/B-to-model mapping and deterministic aggregate calculation.
- `reporting`: typed report assembly and JSON, Markdown, and CSV rendering.
- `runs`: lifecycle state enforcement, atomic local artifacts, environment
  capture, cross-artifact validation, and reproducibility manifests.
- `observability`: structured JSON logging with configured-secret redaction.
- `orchestrator`: the application facade that sequences the independent layers.

Dependencies point inward toward typed contracts. Dataset adapters do not know
about model APIs; the inference engine does not know provider types; judging
does not calculate report metrics; reporting does not make model requests.

## Provider independence

Every model role is described by only:

- `base_url`;
- model `name`;
- optional `api_key`;
- request timeout;
- OpenAI-compatible generation parameters.

The shared client sends non-streaming `POST <base_url>/chat/completions`
requests. LiteLLM proxies, vLLM servers, OpenAI, and internal gateways require
no provider branches in pipeline code as long as they implement this contract.

## Core data contracts

Normalized dataset record:

```json
{"id":"sample-1","prompt":"Explain AI"}
```

Inference response:

```json
{
  "id":"sample-1",
  "prompt":"Explain AI",
  "base_response":"...",
  "fine_tuned_response":"...",
  "metadata":{}
}
```

Judge decision returned by the endpoint:

```json
{
  "winner":"A",
  "scores":{"A":4,"B":3},
  "criteria_scores":{"correctness":{"A":4,"B":3}},
  "reason":"Candidate A is more accurate."
}
```

All persisted contracts are validated before downstream consumption.

## Run lifecycle

The enforced state sequence is:

```text
initialized
  → dataset_prepared
  → inference_running
  → inference_completed | inference_partial
  → evaluation_running
  → evaluation_completed | evaluation_partial
  → reporting
  → completed
```

Resume may re-enter inference from a later or interrupted state, but completion
methods can only run from their corresponding `*_running` state. Reporting is
not allowed without dataset, inference, and evaluation manifests.

## Failure and resume model

- Client retries handle transient HTTP, connection, rate-limit, and timeout
  failures according to configuration.
- A base or fine-tuned failure is stored per role; other samples continue.
- Inference checkpoints are flushed per completed sample.
- Resume skips fully successful inference samples and retries only failed roles.
- Judge API and output-validation failures are stored per sample.
- Resume reuses successful judge results and retries unsuccessful ones.
- Artifact/schema corruption is fatal because continuing could produce an
  incorrect comparison.

## Reproducibility and integrity

Each completed run stores:

- a sanitized resolved configuration snapshot;
- source dataset SHA-256;
- prompt-template SHA-256;
- random seed and runtime settings;
- Python, OS, CPU, and direct dependency versions;
- SHA-256 and size for every core run artifact;
- a cross-artifact validation result.

Before completion, sample IDs, ordering, counts, model metadata, CSV row count,
and all required files are checked across the entire run.

## Deliberate exclusions

The v1 architecture contains no UI, database, plugin framework, distributed
executor, model runtime, cloud deployment, or advanced statistical analysis.

The command flow is configuration load (plus optional local handoff overlay),
base preflight, fine-tuned preflight, judge preflight, run initialization,
dataset preparation, inference, judge evaluation, metrics, and reports. All
preflight calls reuse the production client/parser. A preflight failure is
role-specific and leaves no completed run artifact.

Project #1 integration is file-contract-only. Project #2 owns a small parser for
the current provider-neutral schema and neither imports Project #1 nor queries
its services, database, serving implementation, or gateway internals.
