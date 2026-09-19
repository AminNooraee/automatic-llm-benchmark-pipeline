# Docker deployment

The image is a CPU-only command-line runner. It does not contain a language
model, model weights, CUDA libraries, or an inference server. Base,
fine-tuned, and judge models must be reachable through external
OpenAI-compatible Chat Completions endpoints.

## Build

From the repository root:

```bash
docker build -t automatic-llm-benchmark-pipeline:1.0.0 .
```

The default image uses Python 3.12. Override it with a supported Python version:

```bash
docker build --build-arg PYTHON_VERSION=3.11 \
  -t automatic-llm-benchmark-pipeline:1.0.0 .
```

## Runtime layout

Mount configuration, datasets, prompts, and output separately. A typical
container configuration uses absolute paths such as:

```yaml
dataset:
  path: /data/benchmark.jsonl
evaluation:
  prompt_template: /config/judge.txt
output:
  runs_dir: /output
```

Run validation without contacting model endpoints:

```bash
docker run --rm \
  -v "$PWD/config:/config:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/output:/output" \
  automatic-llm-benchmark-pipeline:1.0.0 \
  validate --config /config/benchmark.yaml
```

Run the benchmark:

```bash
docker run --rm \
  -e BASE_API_KEY \
  -e FINE_TUNED_API_KEY \
  -e JUDGE_API_KEY \
  -v "$PWD/config:/config:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/output:/output" \
  automatic-llm-benchmark-pipeline:1.0.0 \
  run --config /config/benchmark.yaml
```

The host output directory must be writable by the container's unprivileged
`benchmark` user.

## Endpoint connectivity

Use normal HTTPS URLs for remote gateways. When an endpoint runs on the Docker
host, `localhost` inside the container refers to the container itself:

- Docker Desktop supports `host.docker.internal`.
- On Linux, add `--add-host=host.docker.internal:host-gateway`, or use host
  networking when appropriate for the environment.
- For Compose or another container network, use the inference service name.

No inbound port is required because the benchmark is a finite CLI job and only
makes outbound API requests.

## Secrets

Reference environment variables from YAML instead of embedding keys:

```yaml
models:
  base:
    api_key: ${BASE_API_KEY}
```

Do not bake configuration containing credentials into a derived image. API keys
are redacted from configuration snapshots, logs, and reproducibility metadata.

## Resource behavior

The container performs dataset parsing, HTTP requests, JSON processing, and
report generation on CPU. Memory use depends mainly on dataset and response
size. `runtime.concurrency` should be set according to endpoint rate limits and
available memory; the production maximum is 256.

Cloud orchestration, hosted services, bundled models, and GPU deployment are
outside the scope of this image.
