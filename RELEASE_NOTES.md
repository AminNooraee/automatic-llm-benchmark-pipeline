# Release Notes

## v1.0.0 - Automatic LLM Benchmark Pipeline

Release date: 2026-09-19

### Summary

Version 1.0.0 is the first production release of the Automatic LLM Benchmark
Pipeline. It compares a fine-tuned language model with its base model using the
same benchmark prompts, evaluates anonymous response pairs with a configurable
LLM judge, and generates traceable aggregate and sample-level reports.

This release is provider-independent. Base, fine-tuned, and judge models are
accessed exclusively through the OpenAI-compatible Chat Completions contract,
supporting compatible LiteLLM proxies, vLLM servers, the OpenAI API, and
internal company gateways without provider-specific pipeline code.

### Major features

- **Dataset adapters:** JSON, JSONL, CSV, and Parquet ingestion with automatic
  recognition of prompt, question/answer, instruction/input, OpenAI messages,
  ChatML, and ShareGPT schemas, plus explicit custom column mappings.
- **OpenAI-compatible model interface:** one validated and retry-capable client
  contract for base, fine-tuned, and judge endpoints.
- **Base vs fine-tuned comparison:** identical prompt delivery, normalized
  response collection, bounded concurrency, per-role error capture,
  checkpoints, and safe resume.
- **LLM judge evaluation:** configurable criteria and prompt template,
  reproducible randomized A/B answer order, hidden model identity, strict JSON
  parsing, and model-role mapping.
- **Automatic reports:** deterministic metrics and `report.json`,
  `report.md`, and `samples.csv` outputs with aggregate and per-sample evidence.

### Reliability and reproducibility

- Strict YAML configuration and preflight validation.
- Enforced run lifecycle, isolated run directories, and configuration-safe
  resume.
- Structured JSON logging with configured-secret redaction.
- Retry and timeout handling for OpenAI-compatible requests.
- Cross-artifact validation before a run is marked complete.
- Environment capture, configuration fingerprinting, input hashes, prompt
  hashes, artifact SHA-256 digests, and seeded answer randomization.
- CSV formula-injection protection for exported sample text.

### Packaging and operations

- Python 3.11 and 3.12 project metadata.
- Console entry point: `llm-benchmark`.
- CPU-only Dockerfile using a non-root runtime user.
- No model weights, model server, provider SDK, database, or UI bundled in the
  package or image.
- External OpenAI-compatible endpoints remain separately operated.

### Generated artifacts

Each completed run contains:

- the sanitized configuration snapshot and lifecycle manifest;
- original and normalized datasets;
- inference checkpoints and paired responses;
- structured judge results;
- JSON, Markdown, and CSV reports;
- structured logs; and
- environment, artifact-validation, and reproducibility metadata.

### Validation status

- 93 automated tests passed.
- Complete mocked dataset-to-report benchmark passed through local
  OpenAI-compatible HTTP endpoints.
- Required artifact and cross-artifact validation passed.
- Secret-redaction and reproducibility validation passed.
- A clean local v1.0.0 wheel was built successfully during release
  preparation.

### Known limitations

- Docker was unavailable in the local acceptance environment, so the image
  build must be completed in a Docker-enabled release environment.
- Production endpoint validation is environment dependent and must cover local
  network, authentication, quota, model, and gateway policies.
- Human evaluation, distributed execution, database storage, UI, cloud
  deployment, charts, and advanced statistical analysis are outside v1.0.0.

### First-release guidance

This is the first public release, so there are no upgrade or data-migration
steps. Review the [README](README.md), [configuration guide](docs/configuration.md),
and [Project #2 acceptance report](PROJECT_2_ACCEPTANCE_REPORT.md) before
operating a benchmark. Release operators should complete the remaining items
in the [release checklist](RELEASE_CHECKLIST.md) before publishing artifacts
or creating the `v1.0.0` source-control tag.
