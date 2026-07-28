"""Freeze historical checkpoints after validation and before any test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNS = (
    "qm9_model__qm9_domain",
    "qmugs_model__qmugs_domain",
    "qm9_model__combined_domain",
    "qmugs_model__combined_domain",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    acceptance = json.loads(args.acceptance_report.read_text())
    errors = []
    if acceptance.get("status") != "passed":
        errors.append("recovery acceptance did not pass")
    reports = {}
    for run in RUNS:
        path = args.validation_root / run / "report.json"
        if not path.is_file():
            errors.append(f"missing validation report: {path}")
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "passed":
            errors.append(f"validation failed: {run}")
        reports[run] = {
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "metrics": report.get("metrics"),
            "report": str(path.resolve()),
            "report_sha256": _sha256(path),
            "rows": report.get("rows"),
            "split_file_sha256": report.get("split_file_sha256"),
            "status": report.get("status"),
        }

    result = {
        "acceptance_report": str(args.acceptance_report.resolve()),
        "acceptance_report_sha256": _sha256(args.acceptance_report),
        "errors": errors,
        "frozen_models": {
            "QM9_perturbed_fock": "9759da26660c619de9c3bbf4c2dc164343ee90e08e22b3fcdcc9682dacb9bd09",
            "QMUGSBin0_perturbed_fock": "dde9e2e940ebbfcf4c74681b3264c1add71bf3539634e1b81bacffd5bd08be32",
        },
        "policy": (
            "Both predeclared historical checkpoints are frozen if recovery acceptance and all "
            "validation executions are finite/complete. Validation metrics are reported but do "
            "not tune either checkpoint; test remains unread until this artifact exists."
        ),
        "validation_reports": reports,
        "status": "frozen" if not errors else "failed",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
