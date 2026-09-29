# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added

- Default base, fine-tuned, and judge endpoint preflight plus a standalone
  `preflight` command and explicit `--skip-preflight` override.
- Optional Project #1 schema-v1 endpoint handoff adapter.
- Safe one-command CPU Docker launcher with minimal environment forwarding.

### Changed

- Chat Completions parsing now tolerates reasoning metadata while requiring
  final assistant content.
- Judge parsing safely recovers exactly one schema-valid JSON decision from
  fences or prose and rejects ambiguous output.
- Reports explicitly state when no successful judge evaluation supports a
  comparison conclusion.

## 1.0.0 — 2026-09-19

First production release.

### Added

- Strict YAML configuration with environment-based secrets and semantic checks.
- JSON, JSONL, CSV, and Parquet dataset adapters with common schema detection.
- Provider-independent OpenAI-compatible Chat Completions client.
- Resumable paired inference for base and fine-tuned model endpoints.
- Seeded anonymous A/B judge evaluation with strict JSON validation.
- A/B-to-model metrics and JSON, Markdown, and CSV reports.
- Structured redacted logging and enforced run lifecycle transitions.
- Environment capture, cross-artifact validation, and SHA-256 reproducibility
  metadata.
- CPU-only Docker image for external model endpoints.
- Unit, integration, CLI, and complete mocked end-to-end tests.

### Security and reliability

- API keys are excluded from snapshots and redacted from logs and metadata.
- Run artifacts are written atomically and constrained to their run directory.
- CSV text is neutralized against spreadsheet formula execution.
- Resume rejects incompatible configuration fingerprints and missing artifacts.

### Deliberate exclusions

- UI, database, distributed execution, cloud deployment, bundled model serving,
  plugins, human evaluation, charts, and advanced statistical analysis.
