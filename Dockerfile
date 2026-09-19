# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.title="Automatic LLM Benchmark Pipeline" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Provider-independent LLM benchmark runner"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system benchmark \
    && adduser --system --ingroup benchmark --home /home/benchmark benchmark

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install .

COPY configs ./configs
COPY examples ./examples
COPY prompts ./prompts
COPY docs ./docs

RUN mkdir -p /workspace /app/runs \
    && chown -R benchmark:benchmark /workspace /app/runs

USER benchmark
WORKDIR /workspace

ENTRYPOINT ["llm-benchmark"]
CMD ["--help"]
