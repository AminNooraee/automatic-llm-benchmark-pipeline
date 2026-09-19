# Usage examples

## Install for local development

Python 3.11 or 3.12 is recommended:

```bash
python -m venv .venv
```

Activate the environment, then install the package and test dependency:

```bash
python -m pip install -e ".[dev]"
```

## Validate and run

```bash
llm-benchmark validate --config configs/example.yaml
llm-benchmark run --config configs/example.yaml
```

The command is a finite job. Exit code `0` means completion, `2` means an
expected configuration/pipeline error, and `3` means an unexpected internal
error whose details were written to the structured run log.

## Configure secrets

YAML:

```yaml
api_key: ${BASE_API_KEY}
```

POSIX shell:

```bash
export BASE_API_KEY="..."
```

PowerShell:

```powershell
$env:BASE_API_KEY = "..."
```

API keys are not written into the configuration snapshot, environment capture,
or reproducibility manifest.

## Resume a run

```bash
llm-benchmark run \
  --config configs/example.yaml \
  --resume-run runs/<run_id>
```

Use the same resolved configuration. Resume validates the existing layout,
retries failed inference roles, reuses successful judge evaluations, retries
unsuccessful evaluations, and regenerates reports.

## Connect common backends

Only the endpoint details change:

```yaml
models:
  base:
    base_url: http://litellm.internal:4000/v1
    name: base-model
  fine_tuned:
    base_url: http://vllm.internal:8000/v1
    name: fine-tuned-model
  judge:
    base_url: https://company-gateway.example/v1
    name: judge-model
```

Each role still requires `timeout_seconds`; keys and generation parameters are
optional. The pipeline does not need to know which backend serves a role.

## Inspect a completed run

Start with:

- `reports/report.md` for a readable comparison;
- `reports/report.json` for automation;
- `reports/samples.csv` for sample review;
- `manifest.json` for lifecycle and artifact paths;
- `metadata/artifact_validation.json` for integrity status;
- `metadata/reproducibility.json` for hashes and runtime settings;
- `logs/run.jsonl` for structured events.

## Run tests

```bash
python -m pytest
```

The release suite includes unit, integration, CLI, and complete mocked HTTP
end-to-end coverage.

For container usage, see [docker.md](docker.md).
