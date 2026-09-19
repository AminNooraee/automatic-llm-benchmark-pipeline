"""Command-line entry point for the benchmark pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_benchmark import __version__
from llm_benchmark.config.loader import load_config
from llm_benchmark.exceptions import BenchmarkError
from llm_benchmark.observability.logging import configure_logging
from llm_benchmark.orchestrator import BenchmarkPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-benchmark",
        description="Benchmark base and fine-tuned models through OpenAI-compatible APIs.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration without creating a run."
    )
    validate_parser.add_argument("--config", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run dataset preparation, model inference, and judge evaluation."
    )
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume failed inference and evaluation in an existing run directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger: logging.Logger | None = None

    try:
        config = load_config(args.config)
        if args.command == "validate":
            print(f"Configuration is valid: {args.config.resolve()}")
            return 0

        pipeline = BenchmarkPipeline()
        is_resume = args.resume_run is not None
        context = (
            pipeline.open_run(config, args.resume_run)
            if is_resume
            else pipeline.initialize_run(config, args.config)
        )
        logger = configure_logging(
            level=config.runtime.log_level,
            log_file=context.log_path,
            sensitive_values=config.secret_values(),
        )
        logger.info(
            "Run folder initialized",
            extra={
                "event": "run_initialized",
                "run_id": context.run_id,
                "stage": "initialization",
            },
        )
        normalized_path = context.run_dir / "dataset" / "normalized_dataset.jsonl"
        if not is_resume or not normalized_path.is_file():
            dataset_result = pipeline.prepare_dataset(config, context)
            logger.info(
                "Dataset normalized",
                extra={
                    "event": "dataset_prepared",
                    "dataset_format": dataset_result.source_format.value,
                    "schema_format": dataset_result.schema_format.value,
                    "confidence": dataset_result.detection_confidence,
                    "sample_count": dataset_result.sample_count,
                    "run_id": context.run_id,
                    "stage": "dataset",
                },
            )
            print(
                "Dataset prepared: "
                f"{dataset_result.sample_count} samples, "
                f"file_format={dataset_result.source_format.value}, "
                f"schema={dataset_result.schema_format.value}, "
                f"confidence={dataset_result.detection_confidence:.3f}"
            )
        print(f"Run {'resumed' if is_resume else 'initialized'}: {context.run_dir}")
        inference_result = asyncio.run(
            pipeline.run_inference(config, context, resume=is_resume)
        )
        print(
            "Inference completed: "
            f"success={inference_result.successful_samples}, "
            f"partial={inference_result.partial_samples}, "
            f"failed={inference_result.failed_samples}, "
            f"skipped={inference_result.skipped_samples}"
        )
        evaluation_result = asyncio.run(
            pipeline.run_evaluation(config, context, resume=is_resume)
        )
        print(
            "Judge evaluation completed: "
            f"success={evaluation_result.successful_samples}, "
            f"failed={evaluation_result.failed_samples}, "
            f"skipped={evaluation_result.skipped_samples}, "
            f"reused={evaluation_result.reused_samples}"
        )
        report_result = pipeline.generate_reports(config, context)
        print(f"JSON report: {report_result.report_json_path}")
        print(f"Markdown report: {report_result.report_markdown_path}")
        print(f"Sample CSV: {report_result.samples_csv_path}")
        print("Benchmark run completed successfully.")
        return 0
    except BenchmarkError as exc:
        if logger is not None:
            logger.error(
                "Benchmark run failed",
                extra={"event": "run_failed", "stage": "pipeline"},
            )
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        if logger is not None:
            logger.exception(
                "Unexpected benchmark failure",
                extra={"event": "run_failed", "stage": "pipeline"},
            )
        print(
            "Unexpected internal error. Inspect the run log for details.",
            file=sys.stderr,
        )
        return 3
