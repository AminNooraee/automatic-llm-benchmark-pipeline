from __future__ import annotations

import csv
import json

import pytest

from llm_benchmark.config.loader import load_config
from llm_benchmark.evaluation.artifact_store import JudgeArtifactStore
from llm_benchmark.evaluation.models import (
    AnswerOrder,
    AnswerScores,
    EvaluatedModel,
    JudgeEvaluationStatus,
    JudgeResult,
    JudgeResultMetadata,
    JudgeRunResult,
    JudgeWinner,
)
from llm_benchmark.domain.runs import RunManifest, RunStatus
from llm_benchmark.inference.models import InferenceRunResult
from llm_benchmark.orchestrator import BenchmarkPipeline
from llm_benchmark.runs.artifact_store import LocalArtifactStore
from llm_benchmark.runs.manager import RunManager
from llm_benchmark.runs.validator import ArtifactValidationError, RunArtifactValidator


def test_full_judge_results_to_report_flow(config_factory) -> None:
    config_path = config_factory()
    config = load_config(config_path)
    run_manager = RunManager()
    pipeline = BenchmarkPipeline(run_manager=run_manager)
    context = pipeline.initialize_run(config, config_path)
    pipeline.prepare_dataset(config, context)
    inference_path = LocalArtifactStore(context.run_dir).write_jsonl(
        "inference/responses.jsonl",
        [
            {
                "id": "example-1",
                "prompt": "Example prompt",
                "base_response": "Base response",
                "fine_tuned_response": "Fine-tuned response",
            }
        ],
    ).path
    checkpoint_path = LocalArtifactStore(context.run_dir).write_jsonl(
        "inference/checkpoints.jsonl",
        [{"checkpoint": "test"}],
    ).path
    run_manager.mark_inference_started(context)
    run_manager.mark_inference_finished(
        context,
        InferenceRunResult(
            artifact_path=inference_path,
            checkpoint_path=checkpoint_path,
            total_samples=1,
            successful_samples=1,
            partial_samples=0,
            failed_samples=0,
            skipped_samples=0,
            attempted_samples=1,
        ),
    )
    order = AnswerOrder(
        A=EvaluatedModel.FINE_TUNED,
        B=EvaluatedModel.BASE,
    )
    judge_result = JudgeResult(
        id="example-1",
        prompt="Example prompt",
        winner=JudgeWinner.A,
        scores=AnswerScores(A=5, B=2),
        criteria_scores={
            "correctness": AnswerScores(A=5, B=2),
            "clarity": AnswerScores(A=5, B=3),
        },
        reason="Candidate A is better.",
        metadata=JudgeResultMetadata(
            status=JudgeEvaluationStatus.SUCCESS,
            answer_order=order,
            resolved_winner=order.resolve(JudgeWinner.A),
            judge_model="judge-model",
            attempts=1,
        ),
    )
    judge_path = JudgeArtifactStore(context.run_dir).write_results([judge_result])
    run_manager.mark_evaluation_started(context)
    run_manager.mark_evaluation_finished(
        context,
        JudgeRunResult(
            artifact_path=judge_path,
            total_samples=1,
            successful_samples=1,
            failed_samples=0,
            skipped_samples=0,
            reused_samples=0,
            attempted_samples=1,
        ),
    )

    result = pipeline.generate_reports(config, context)

    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert report["aggregate_metrics"]["fine_tuned_model_wins"] == 1
    assert report["aggregate_metrics"]["average_scores_per_model"] == {
        "base_model": 2.0,
        "fine_tuned_model": 5.0,
    }
    assert report["per_sample_results"][0]["winner"] == "fine_tuned"
    with result.samples_csv_path.open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["winner"] == "fine_tuned"
    assert csv_row["base_response"] == "Base response"
    assert csv_row["fine_tuned_response"] == "Fine-tuned response"

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["reporting"]["report_json"] == "reports/report.json"
    assert manifest["reporting"]["report_markdown"] == "reports/report.md"
    assert manifest["reporting"]["samples_csv"] == "reports/samples.csv"
    assert manifest["reporting"]["artifact_validation"] == (
        "metadata/artifact_validation.json"
    )
    assert manifest["reporting"]["reproducibility"] == (
        "metadata/reproducibility.json"
    )

    result.samples_csv_path.write_text(
        "sample_id,prompt,base_response,fine_tuned_response,evaluation_status,"
        "winner,base_score,fine_tuned_score,criteria_scores,judge_reason,"
        "error_message\n",
        encoding="utf-8",
    )
    reporting_manifest = RunManifest.model_validate(manifest).model_copy(
        update={"status": RunStatus.REPORTING}
    )
    with pytest.raises(ArtifactValidationError, match="CSV sample count"):
        RunArtifactValidator().validate_and_write(
            config=config,
            context=context,
            manifest=reporting_manifest,
            report_result=result,
        )
