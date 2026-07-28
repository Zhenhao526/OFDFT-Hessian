"""Record and verify the frozen validation-to-Test Slurm execution order."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


JOBS = {
    696: "initial_final_density_validation_forward",
    697: "initial_final_density_validation_merge",
    700: "initial_validation_freeze_before_any_test_preprocessing",
    701: "test_identity_manifest_after_initial_freeze",
    702: "test_final_density_cache",
    703: "test_final_density_cache_merge",
    786: "validation_scf6_cache",
    787: "validation_scf6_cache_merge",
    788: "validation_dual_density_forward",
    789: "validation_dual_density_merge",
    790: "freeze_validation_models",
    791: "test_scf6_cache_after_freeze",
    792: "test_scf6_cache_merge",
    793: "one_shot_test_dual_density_forward_primary_array",
    918: "qm9_combined_test_batch1_oom_recovery",
    926: "one_shot_test_dual_density_merge_after_recovery",
}
EDGES = tuple(zip(JOBS, tuple(JOBS)[1:]))
CANCELLED = (704, 705, 761, 794, 795, 796)
PRIMARY_TEST_JOB = 793
PRIMARY_FAILED_TASKS = set(range(16, 24))


def _timestamp(value: str | None) -> datetime | None:
    if value in (None, "", "Unknown", "None"):
        return None
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    requested = ",".join(map(str, (*JOBS, *CANCELLED)))
    fields = "JobIDRaw,JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList"
    command = [
        "sacct",
        "-j",
        requested,
        "--starttime",
        "2026-07-15",
        "-X",
        "-n",
        "-P",
        "-o",
        fields,
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    names = fields.split(",")
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if values and values[-1] == "":
            values.pop()
        row = dict(zip(names, values, strict=True))
        raw = row["JobID"]
        try:
            row["parent_job_id"] = int(raw.split("_", 1)[0].split(".", 1)[0])
        except ValueError:
            continue
        rows.append(row)
    errors = []
    stages = {}
    for job_id, purpose in JOBS.items():
        selected = [row for row in rows if row["parent_job_id"] == job_id]
        if not selected:
            errors.append(f"missing sacct rows for job {job_id}")
            continue
        if job_id == PRIMARY_TEST_JOB:
            task_rows = [row for row in selected if "_" in row["JobID"]]
            failed_tasks = {
                int(row["JobID"].split("_", 1)[1])
                for row in task_rows
                if row["State"].startswith("FAILED")
            }
            completed_tasks = {
                int(row["JobID"].split("_", 1)[1])
                for row in task_rows
                if row["State"].startswith("COMPLETED")
            }
            expected_completed = set(range(32)) - PRIMARY_FAILED_TASKS
            if failed_tasks != PRIMARY_FAILED_TASKS:
                errors.append(
                    f"job {job_id} failed tasks {sorted(failed_tasks)} != "
                    f"{sorted(PRIMARY_FAILED_TASKS)}"
                )
            if completed_tasks != expected_completed:
                errors.append(
                    f"job {job_id} completed tasks {sorted(completed_tasks)} != "
                    f"{sorted(expected_completed)}"
                )
        else:
            bad = [
                row["State"]
                for row in selected
                if not row["State"].startswith("COMPLETED")
            ]
            if bad:
                errors.append(f"job {job_id} non-completed states: {bad}")
        starts = [_timestamp(row["Start"]) for row in selected]
        ends = [_timestamp(row["End"]) for row in selected]
        starts = [value for value in starts if value is not None]
        ends = [value for value in ends if value is not None]
        stages[str(job_id)] = {
            "purpose": purpose,
            "rows": selected,
            "start": min(starts).isoformat() if starts else None,
            "end": max(ends).isoformat() if ends else None,
        }
        if job_id == PRIMARY_TEST_JOB:
            stages[str(job_id)]["accepted_completed_tasks"] = sorted(completed_tasks)
            stages[str(job_id)]["failed_no_summary_tasks"] = sorted(failed_tasks)
    order_checks = []
    for upstream, downstream in EDGES:
        before = _timestamp(stages.get(str(upstream), {}).get("end", ""))
        after = _timestamp(stages.get(str(downstream), {}).get("start", ""))
        passed = before is not None and after is not None and before <= after
        order_checks.append(
            {
                "upstream": upstream,
                "downstream": downstream,
                "upstream_end": before.isoformat() if before else None,
                "downstream_start": after.isoformat() if after else None,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"invalid or unproven order {upstream}->{downstream}")
    cancelled = {
        str(job_id): [row for row in rows if row["parent_job_id"] == job_id]
        for job_id in CANCELLED
    }
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "definition": (
            "Slurm accounting proof for the serial validation-freeze-Test funnel. Job 700 freezes "
            "the initial final-density validation before any Test preprocessing. Job 790 refreezes "
            "the complete dual-density validation before Test SCF-6 preprocessing and before the "
            "only accepted Test forward. Primary Test job 793 completed 24 shards; tasks 16-23 "
            "produced no summary after batch-size-64 A100 OOM. Job 918 reruns exactly those eight "
            "model/domain shards at batch size 1 without changing model, samples, SCF selectors, "
            "dtype or metrics; job 926 merges the complete accepted rows."
        ),
        "sacct_command": command,
        "stages": stages,
        "order_checks": order_checks,
        "cancelled_non_scientific_attempts": cancelled,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
