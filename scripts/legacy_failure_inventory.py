"""Freeze failed/cancelled recovery attempts without treating them as science."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts(root: Path, relative: str) -> list[dict]:
    path = root / relative
    paths = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    return [
        {
            "bytes": item.stat().st_size,
            "path": str(item.resolve()),
            "sha256": _sha256(item),
        }
        for item in paths
        if item.is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.restore_root.resolve()
    attempts = [
        {
            "id": "content_merge_noncovering_index_job575",
            "state": "cancelled",
            "cause": "SQLite merge used non-covering random reads and did not finish at full scale.",
            "resolution": "Rebuilt a covering index; replacement merge job 664 passed.",
            "path": "content_audit_full_20260715/merge_failed_noncovering_index.log",
        },
        {
            "id": "acceptance_batch_attempt1_jobs642_643",
            "state": "failed",
            "cause": "Variable-size square preprocessing tensors were included in PyG batching.",
            "resolution": "Excluded consumed overlap matrices from model-input batching.",
            "path": "model_acceptance_20260715/failed_attempt1",
        },
        {
            "id": "acceptance_batch_attempt2_jobs648_649",
            "state": "failed",
            "cause": "The single-sample diagnostic tried to convert a three-graph batch error to a scalar.",
            "resolution": "Separated per-sample diagnostics from batched consistency checks.",
            "path": "model_acceptance_20260715/failed_attempt2",
        },
        {
            "id": "acceptance_absolute_energy_gate_job662",
            "state": "failed_gate",
            "cause": "A fixed absolute tolerance was inappropriate for extensive total energy.",
            "resolution": "Predeclared relative and per-electron float32 gates; checkpoint outputs were not changed.",
            "path": "model_acceptance_20260715/report_failed_extensive_absolute_tolerance.json",
        },
        {
            "id": "baseline_calibration_variable_matrix_batch",
            "state": "failed",
            "cause": "Calibration loader attempted to concatenate variable-size overlap matrices.",
            "resolution": "Excluded preprocessing-only matrices in the production evaluator.",
            "path": "baseline_calibration_20260715/failed_attempt1.log",
        },
        {
            "id": "raw_validation_transform_job667",
            "state": "cancelled",
            "cause": "Four model/domain runs redundantly rebuilt expensive natural-representation transforms and produced no first batch.",
            "resolution": "Created read-only-source transformed caches and verified raw-to-cache equivalence.",
            "path": "scientific_baseline_validation_20260715/failed_raw_transform_attempt",
        },
        {
            "id": "remote_total_derivative_unit_job663",
            "state": "failed_preflight",
            "cause": "Recovery-server code snapshot does not contain the new total-derivative test modules.",
            "resolution": "Ran the candidate implementation locally (25 passed) and explicitly left remote deployment pending code-release audit.",
            "path": "total_derivative_unit_20260715.log",
        },
        {
            "id": "test_cache_gpu_transform_smoke_job760",
            "state": "cancelled",
            "cause": "The largest 216-atom sample remained CPU/PySCF integral-build bound before reaching the GPU eigensolver.",
            "resolution": "Retained the CPU-sharded cache path; no scientific metric was produced.",
            "path": "slurm-760.out",
        },
        {
            "id": "final_only_test_jobs704_705",
            "state": "cancelled_before_execution",
            "cause": "The full-split final-density-only difference target is identically zero and was not adequate scientific coverage.",
            "resolution": "Cancelled before any forward pass; replaced by frozen SCF-6 plus final-density Test jobs.",
            "path": None,
        },
        {
            "id": "concurrent_scf6_cache_job761",
            "state": "cancelled_before_sample_output",
            "cause": "Concurrent raw transform arrays contended on the shared filesystem: about 20 minutes elapsed, about 7 CPU seconds accrued, and zero samples were written.",
            "resolution": "Preserved sacct/task logs and resubmitted the cache/baseline/Test funnel serially after the final-density cache.",
            "path": "baseline_cache_validation_scf6_20260715/failed_concurrent_io_attempt_job761",
        },
        {
            "id": "superseded_final_report_job795",
            "state": "cancelled_before_execution",
            "cause": "Slurm had frozen an earlier final-report wrapper before immutability and completion-audit stages were added.",
            "resolution": "Cancelled at zero elapsed time and resubmitted the current wrapper as job 796 with the same afterok:794 scientific dependency.",
            "path": None,
        },
        {
            "id": "test_qm9_combined_batch64_oom_job793_tasks16_23",
            "state": "failed_infrastructure_no_summary",
            "cause": (
                "All eight qm9_model__combined_domain shards exceeded one 80-GB A100 at "
                "batch_size=64 on the large QMUGS Test graphs; no shard summary or accepted row "
                "artifact was produced."
            ),
            "resolution": (
                "Archived the OOM logs and reran only the eight missing shards with the same "
                "frozen checkpoint, split, SCF selectors, float32 dtype, cache and metrics at "
                "batch_size=1. Computational batching is not a model or metric change."
            ),
            "path": (
                "scientific_baseline_test_full_20260715/"
                "qm9_model__combined_domain/failed_batch64_job793"
            ),
        },
    ]
    errors = []
    for attempt in attempts:
        relative = attempt.pop("path")
        if relative is None:
            attempt["artifacts"] = []
        else:
            attempt["artifacts"] = _artifacts(root, relative)
            if not attempt["artifacts"]:
                errors.append(f"missing retained artifact for {attempt['id']}: {relative}")
        attempt["used_as_scientific_result"] = False
    report = {
        "status": "audit_complete" if not errors else "failed",
        "errors": errors,
        "definition": (
            "Infrastructure, harness, and protocol attempts that did not yield accepted scientific "
            "results. Logs are retained; replacements are linked by the resolution field."
        ),
        "attempts": attempts,
        "attempt_count": len(attempts),
        "retained_file_count": sum(len(item["artifacts"]) for item in attempts),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
