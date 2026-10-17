from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from llm_benchmark.domain.runs import RunContext
from llm_benchmark.evaluation.models import JudgeRunResult
from llm_benchmark.inference.models import InferenceRunResult
from llm_benchmark.orchestrator import BenchmarkPipeline


def _config():
    return SimpleNamespace(
        models=SimpleNamespace(base=object(), fine_tuned=object(), judge=object()),
        runtime=SimpleNamespace(
            concurrency=2,
            random_seed=7,
            max_retries=1,
            retry_backoff_seconds=0,
        ),
        evaluation=SimpleNamespace(prompt_template="prompt.txt", criteria=[]),
    )


def _context(tmp_path):
    return RunContext(
        run_id="run-id",
        run_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        config_snapshot_path=tmp_path / "config.yaml",
        log_path=tmp_path / "run.log",
    )


def test_inference_and_judge_wall_clock_timing_is_captured(tmp_path) -> None:
    manager = Mock()
    pipeline = BenchmarkPipeline(run_manager=manager)
    client = Mock()
    inference_result = InferenceRunResult(
        artifact_path=tmp_path / "responses.jsonl",
        checkpoint_path=tmp_path / "checkpoints.jsonl",
        total_samples=1,
        successful_samples=1,
        partial_samples=0,
        failed_samples=0,
        skipped_samples=0,
        attempted_samples=1,
    )
    judge_result = JudgeRunResult(
        artifact_path=tmp_path / "judge.jsonl",
        total_samples=1,
        successful_samples=1,
        failed_samples=0,
        skipped_samples=0,
        reused_samples=0,
        attempted_samples=1,
    )

    with (
        patch("llm_benchmark.orchestrator.InferenceEngine") as inference_engine,
        patch("llm_benchmark.orchestrator.JudgeEngine") as judge_engine,
        patch(
            "llm_benchmark.orchestrator.time.perf_counter",
            side_effect=[100.0, 104.5, 200.0, 207.25],
        ),
    ):
        inference_engine.return_value.run = AsyncMock(return_value=inference_result)
        judge_engine.return_value.run = AsyncMock(return_value=judge_result)
        asyncio.run(
            pipeline.run_inference(
                _config(), _context(tmp_path), resume=False, model_client=client
            )
        )
        asyncio.run(
            pipeline.run_evaluation(
                _config(), _context(tmp_path), resume=False, model_client=client
            )
        )

    assert pipeline._inference_duration == 4.5
    assert pipeline._judge_duration == 7.25
    manager.mark_inference_finished.assert_called_once_with(
        _context(tmp_path), inference_result
    )
    manager.mark_evaluation_finished.assert_called_once_with(
        _context(tmp_path), judge_result
    )
