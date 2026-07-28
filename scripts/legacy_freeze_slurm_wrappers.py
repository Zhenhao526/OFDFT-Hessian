"""Preserve the exact batch scripts frozen by Slurm for the final recovery funnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


JOBS = (
    702,
    703,
    786,
    787,
    788,
    789,
    790,
    791,
    792,
    793,
    794,
    796,
    918,
    926,
    927,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    errors = []
    for job_id in JOBS:
        path = args.output_dir / f"job_{job_id}.sbatch"
        temporary = args.output_dir / f".job_{job_id}.controller.tmp"
        temporary.unlink(missing_ok=True)
        result = subprocess.run(
            ["scontrol", "write", "batch_script", str(job_id), str(temporary)],
            capture_output=True,
            text=True,
        )
        controller_ok = (
            result.returncode == 0
            and temporary.is_file()
            and "retrieval failed" not in result.stderr.lower()
        )
        if controller_ok:
            temporary.replace(path)
            capture = "slurm_controller_current"
        elif path.is_file():
            temporary.unlink(missing_ok=True)
            capture = "retained_prior_slurm_controller_capture"
        else:
            errors.append(
                {
                    "job_id": job_id,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                }
            )
            continue
        artifacts[str(job_id)] = {
            "bytes": path.stat().st_size,
            "capture": capture,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    report = {
        "status": "passed" if not errors and len(artifacts) == len(JOBS) else "failed",
        "errors": errors,
        "definition": (
            "Exact batch wrappers stored by the Slurm controller at sbatch time. This prevents "
            "later workspace edits from being mistaken for the configuration that actually ran."
        ),
        "jobs": artifacts,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
