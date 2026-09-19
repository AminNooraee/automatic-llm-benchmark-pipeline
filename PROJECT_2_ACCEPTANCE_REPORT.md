# Project #2 Acceptance Report

**Project:** Automatic LLM Benchmark Pipeline  
**Release:** v1.0.0  
**Acceptance scope:** Core benchmark pipeline, production hardening, and
release documentation  
**Status:** Implementation complete and ready for company review and technical
handover

## Project Overview

### Goal

Project #2 provides a reusable, reproducible way to determine whether a
fine-tuned language model improves on its base model. It runs the same
benchmark prompts against both endpoints, uses a separate LLM judge to compare
anonymous answers, converts the judge's A/B decisions back to the real model
roles, and generates machine-readable and human-readable evidence.

### Problem being solved

Fine-tuning produces a model artifact or endpoint, but production acceptance
requires more than confirming that training completed. Teams need a consistent
comparison that:

- gives the base and fine-tuned models identical input;
- avoids provider-specific integrations;
- reduces judge bias by hiding model identity;
- preserves sample-level evidence and failure information;
- produces repeatable metrics and reviewable reports; and
- records enough configuration, environment, and artifact metadata for audit
  and maintenance.

This project standardizes that evaluation boundary.

### Relationship with Project #1

The two projects form a connected workflow without sharing implementation
code:

```text
Project #1
Dataset -> Fine-tuning -> Fine-tuned model endpoint

Project #2
Base model endpoint + Fine-tuned model endpoint -> Benchmark -> Report
```

Project #1 is responsible for creating or serving the fine-tuned model.
Project #2 only consumes endpoints. A training framework, model family, or
serving backend can change without coupling the benchmark pipeline to the
fine-tuning system.

## Architecture Overview

### Complete pipeline

```text
Dataset
   |
   v
Dataset Adapter
   |
   v
Normalized Benchmark Data ({id, prompt})
   |
   +-----------------------+
   |                       |
   v                       v
Base Model            Fine-tuned Model
   |                       |
   +-----------+-----------+
               |
               v
        Inference Engine
               |
               v
          Judge Model
               |
               v
            Metrics
               |
               v
            Reports
```

The configuration layer validates the dataset, three model roles, judge prompt,
criteria, runtime behavior, and output location before a run begins. The run
manager then enforces lifecycle transitions and stores each layer's artifacts.
Structured logs, artifact validation, environment capture, and reproducibility
metadata support operational review without changing the benchmark decision
logic.

### OpenAI-compatible API design

Base, fine-tuned, and judge calls all go through one provider-independent model
client. Each role supplies only:

- `base_url`;
- model `name`;
- optional `api_key`;
- `timeout_seconds`; and
- OpenAI-compatible `generation_parameters`.

The client sends a non-streaming Chat Completions request to
`POST <base_url>/chat/completions` and parses the standard assistant message
content. Dataset, inference, evaluation, metrics, and reporting components do
not know which vendor or serving technology is behind an endpoint.

### Provider independence

Provider independence is achieved by depending on the OpenAI-compatible HTTP
contract rather than a provider SDK. The pipeline has no LiteLLM, vLLM, OpenAI,
or internal-gateway branches. These systems connect by exposing a compatible
Chat Completions endpoint and supplying the appropriate URL, model identifier,
credentials, timeout, and generation parameters:

| Backend | Connection approach |
| --- | --- |
| LiteLLM proxy | Configure the proxy's OpenAI-compatible `/v1` URL and routed model name |
| vLLM | Configure the vLLM OpenAI-compatible server URL and served model name |
| OpenAI API | Configure the OpenAI API base URL, model name, and API key |
| Internal company gateway | Configure its compatible base URL, gateway model identifier, and optional credential |

Endpoint-specific authentication policies, routing, quotas, and availability
remain responsibilities of the target environment.

## Supported Capabilities

### Dataset support

Supported file formats:

- JSON
- JSONL
- CSV
- Parquet

Supported record schemas:

- simple prompt records;
- question/answer records, with `question` normalized into the prompt;
- instruction records, with `instruction` and optional `input` combined;
- OpenAI `messages` conversations;
- structured or serialized ChatML;
- ShareGPT conversations; and
- custom prompt and optional ID column mappings, including dotted paths.

All accepted records become:

```json
{"id": "sample-000001", "prompt": "The prompt sent to both models"}
```

The adapter layer detects file and record formats, rejects ambiguous schemas,
and reports malformed rows, empty prompts, duplicate IDs, unsupported formats,
and missing configured fields with source context.

### Model support

Three logical endpoints are configured independently:

- base model endpoint;
- fine-tuned model endpoint; and
- judge model endpoint.

Each endpoint uses the same OpenAI-compatible model-client contract. Different
compatible backends may be used for different roles in one run.

### Evaluation

The evaluation layer provides:

- anonymous A/B comparison;
- seeded randomized answer ordering;
- a configurable plain-text judge prompt;
- configurable criteria, weights, and score ranges;
- strict structured JSON validation;
- per-sample API and output-error capture; and
- private A/B-to-model mapping for downstream metrics.

The judge is required to return:

```json
{
  "winner": "A",
  "scores": {"A": 5, "B": 3},
  "criteria_scores": {
    "correctness": {"A": 5, "B": 3}
  },
  "reason": "Candidate A is more accurate."
}
```

## End-to-End Workflow

1. **Load configuration.** Parse strict YAML, expand configured environment
   references, resolve relative paths, validate model and runtime settings, and
   verify the judge prompt and criteria.
2. **Prepare the dataset.** Select the source and file adapter, detect or apply
   the record schema, validate records, generate deterministic IDs where
   needed, and write normalized JSONL.
3. **Send prompts to the base model.** Submit each normalized prompt through
   the shared OpenAI-compatible client and retain response or error metadata.
4. **Send prompts to the fine-tuned model.** Submit the exact same prompt
   through the same client contract using the fine-tuned role's configuration.
5. **Collect responses.** Persist ordered base/fine-tuned response pairs and
   per-role status metadata, with checkpoints that support safe resume.
6. **Send answers to the judge model.** Randomly assign answers to A and B,
   render the configured evaluation prompt, call the judge endpoint, validate
   structured output, and retain the private mapping.
7. **Calculate metrics.** Resolve winners and scores from A/B to base and
   fine-tuned roles, then calculate counts, rates, and average scores.
8. **Generate reports.** Render the same typed results to `report.json`,
   `report.md`, and `samples.csv`, validate cross-artifact consistency, and
   record reproducibility metadata.

## Configuration

Configuration is divided into the following sections:

- `dataset`: input path, file format, and optional custom column mappings.
- `models`: independent `base`, `fine_tuned`, and `judge` endpoint
  configurations. The judge model is a role under this section rather than a
  provider-specific component.
- `evaluation`: judge prompt template and weighted evaluation criteria.
- `runtime`: concurrency, retries, retry backoff, random seed, and log level.
- `output`: the parent directory for immutable run folders.

A root-level configuration can look like:

```yaml
version: 1

dataset:
  path: examples/datasets/benchmark.jsonl
  format: auto
  # Optional custom mapping:
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

The checked-in [example configuration](configs/example.yaml) is ready to copy
and edit. Its local endpoint values are placeholders. Configuration should be
validated before execution:

```bash
llm-benchmark validate --config configs/example.yaml
```

Configured API keys may reference environment variables and are redacted from
persisted snapshots, logs, environment capture, and reproducibility metadata.
See the [configuration guide](docs/configuration.md) for the complete schema.

## Output Artifacts

A new benchmark creates a unique `runs/<run_id>/` directory; an explicit
resume safely continues a compatible existing run:

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

Artifact responsibilities:

- `dataset/original_dataset.*` preserves the source bytes used by the run.
- `dataset/normalized_dataset.jsonl` is the canonical inference input.
- `inference/checkpoints.jsonl` supports per-sample recovery.
- `inference/responses.jsonl` stores base and fine-tuned responses with status
  metadata.
- `evaluation/judge_results.jsonl` stores validated judge decisions, failures,
  answer order, and resolved winner metadata.
- `reports/report.json` contains benchmark, model, dataset, criteria, metrics,
  and per-sample data for automation.
- `reports/report.md` contains the human-readable benchmark review.
- `reports/samples.csv` contains flat sample-level comparison evidence.
- `manifest.json` records lifecycle status and canonical artifact paths.
- `metadata/environment.json` records runtime environment information.
- `metadata/artifact_validation.json` records integrity and cross-artifact
  validation outcomes.
- `metadata/reproducibility.json` records the pipeline version, random seed,
  configuration fingerprint, input hashes, and output hashes.
- `logs/run.jsonl` contains structured, secret-redacted operational events.

## Testing and Validation

The v1.0.0 acceptance baseline is:

- **93 tests passed** across unit, integration, CLI, release, and end-to-end
  suites.
- **End-to-end mocked benchmark completed**, covering dataset preparation,
  base inference, fine-tuned inference, judge evaluation, model-role mapping,
  metrics, and all three report formats.
- **Artifact validation passed**, including required files, schemas, sample
  identifiers, order, counts, report consistency, and secret redaction.
- **Reproducibility validation passed**, including version, random seed,
  configuration fingerprint, and SHA-256 artifact digests.

The complete mocked test uses real local HTTP request/response behavior against
mock OpenAI-compatible endpoints, so it validates integration boundaries
without depending on external providers.

## Limitations

- The Docker image build was not executed locally because Docker was
  unavailable. The Dockerfile and container documentation are included and
  require validation in a Docker-enabled release environment.
- Real production endpoint validation remains environment dependent, including
  network access, authentication policy, gateway compatibility, quotas, model
  availability, and response behavior.
- Human evaluation is not included.
- Distributed execution is not included.
- Database storage and a user interface are not included.
- Advanced statistical analysis, confidence intervals, significance tests, and
  charts are not included.
- The pipeline does not train, download, package, or serve models.

## Future Extensions

Potential follow-on work includes:

- an operational and benchmark-comparison dashboard;
- distributed benchmarking for larger datasets and endpoint fleets;
- additional evaluation strategies, including reference-based metrics,
  multi-judge consensus, and domain-specific rubrics;
- a human feedback and adjudication loop;
- a benchmark database for longitudinal model and run comparison;
- advanced statistical analysis and confidence reporting; and
- organization-specific release and staging integrations.

These extensions can build on the current normalized dataset, model-client,
evaluation, and report contracts without coupling Project #2 to Project #1.

## Acceptance Conclusion

The Project #2 v1.0.0 implementation satisfies the approved core scope:
provider-independent paired inference, blind LLM judging, deterministic metric
mapping, report generation, lifecycle controls, reproducibility metadata,
documentation, and mocked end-to-end validation. No UI, database, distributed
runtime, human evaluation, cloud deployment, or advanced-statistics scope has
been added.

The project is ready for stakeholder review, technical handover, and the
separate GitHub release process after approval.
