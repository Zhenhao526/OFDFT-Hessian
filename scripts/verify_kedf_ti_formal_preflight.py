#!/usr/bin/env python3
"""Verify and checksum a prepared formal KEDF TI production grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.prepare_kedf_ti_formal_from_merged import parse_tau_overrides


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_input(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line == "INPUT_PARAMETERS" or line.startswith("#"):
            continue
        key, separator, value = line.partition(" ")
        if separator:
            values[key] = value.strip()
    return values


def parse_source_step_overrides(
    values: list[str],
) -> dict[tuple[str, str], int]:
    overrides: dict[tuple[str, str], int] = {}
    for value in values:
        key, separator, step_text = value.partition("=")
        phase, label_separator, label = key.partition(":")
        if not separator or not label_separator or phase not in {"solid", "liquid"}:
            raise ValueError(
                "--source-step-override must use "
                "solid:lambda_LABEL=STEP or liquid:lambda_LABEL=STEP"
            )
        step = int(step_text)
        if step < 0:
            raise ValueError("source step override must be non-negative")
        overrides[(phase, label)] = step
    return overrides


def pair_models_by_phase(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    records = manifest.get("pair_models_by_phase")
    if records is not None:
        if set(records) != {"solid", "liquid"}:
            raise ValueError(
                "pair_models_by_phase must contain solid and liquid"
            )
        return records
    return {
        phase: {
            "path": manifest["pair_model"],
            "sha256": manifest["pair_model_sha256"],
        }
        for phase in ("solid", "liquid")
    }


def verify(
    root: Path,
    *,
    target_kedf: str,
    steps: int,
    ranks: int,
    source_step: int,
    minimum_nn: float,
    default_tau: float,
    tau_overrides: dict[tuple[str, str], float],
    source_step_overrides: dict[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "formal_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    phase_pair_models = pair_models_by_phase(manifest)
    merged = Path(manifest["merged_pilot_analysis"])
    sources = {
        (record["phase"], record["label"]): record
        for record in manifest["sources"]
    }
    source_step_overrides = source_step_overrides or {}

    top_checks = {
        "manifest_prepared": manifest.get("status") == "prepared",
        "target_kedf_matches": manifest.get("target_kedf") == target_kedf,
        "steps_match": manifest.get("steps_per_window") == steps,
        "ranks_match": manifest.get("mpi_ranks_per_window") == ranks,
        "pair_sha_matches": all(
            Path(record["path"]).is_file()
            and sha256(Path(record["path"])) == record["sha256"]
            for record in phase_pair_models.values()
        ),
        "merged_sha_matches": (
            merged.is_file()
            and sha256(merged) == manifest.get("merged_pilot_analysis_sha256")
        ),
        "window_count_18": len(sources) == 18,
    }

    windows: list[dict[str, Any]] = []
    checksum_paths = [manifest_path]
    for phase in ("solid", "liquid"):
        expected_pair = phase_pair_models[phase]
        phase_manifest_path = root / phase / "manifest.json"
        phase_manifest = json.loads(phase_manifest_path.read_text())
        checksum_paths.append(phase_manifest_path)
        for window in phase_manifest["windows"]:
            label = window["label"]
            point = root / phase / label
            metadata_path = point / "metadata.json"
            input_path = point / "INPUT"
            run_path = point / "run_local.sh"
            metadata = json.loads(metadata_path.read_text())
            input_values = parse_input(input_path)
            source = sources[(phase, label)]
            source_structure = Path(source["source_structure"])
            expected_tau = tau_overrides.get((phase, label), default_tau)
            expected_source_step = source_step_overrides.get(
                (phase, label), source_step
            )
            old_outputs = list(point.glob("OUT.*"))
            run_text = run_path.read_text()
            checks = {
                "source_step_matches": (
                    source.get("source_step") == expected_source_step
                    and metadata.get("source_step") == expected_source_step
                ),
                "velocities_preserved": (
                    metadata.get("source_velocities_discarded") is False
                ),
                "source_phase_verified": (
                    source.get("source_phase_status") == f"{phase}_verified"
                ),
                "source_nn_gt_minimum": (
                    float(source.get("source_minimum_nearest_neighbor_A", 0.0))
                    > minimum_nn
                ),
                "source_sha_matches": (
                    source_structure.is_file()
                    and sha256(source_structure)
                    == source.get("source_structure_sha256")
                    == metadata.get("source_structure_sha256")
                ),
                "target_kedf_matches": (
                    metadata.get("target_kedf") == target_kedf
                    and input_values.get("of_kinetic") == target_kedf
                ),
                "steps_match": (
                    metadata.get("steps") == steps
                    and input_values.get("md_nstep") == str(steps)
                ),
                "ranks_match": (
                    metadata.get("mpi_ranks") == ranks
                    and f"-np {ranks} " in run_text
                ),
                "tau_matches": (
                    float(metadata.get("csvr_tau", -1.0)) == expected_tau
                    and float(input_values.get("md_csvr_tau", -1.0))
                    == expected_tau
                ),
                "pair_sha_matches": (
                    metadata.get("pair_model_sha256")
                    == expected_pair["sha256"]
                    and metadata.get("pair_model")
                    == expected_pair["path"]
                ),
                "no_old_out": not old_outputs,
                "run_local_executable": (
                    run_path.is_file()
                    and os.access(run_path, os.X_OK)
                    and "abacus_pw_para" in run_text
                ),
            }
            windows.append(
                {
                    "phase": phase,
                    "label": label,
                    "source_step": source.get("source_step"),
                    "source_phase_status": source.get("source_phase_status"),
                    "source_minimum_nearest_neighbor_A": source.get(
                        "source_minimum_nearest_neighbor_A"
                    ),
                    "csvr_tau": expected_tau,
                    "checks": checks,
                }
            )
            checksum_paths.extend(
                point / name for name in ("INPUT", "KPT", "metadata.json", "STRU")
            )

    all_checks = list(top_checks.values())
    all_checks.extend(
        passed for window in windows for passed in window["checks"].values()
    )
    verified = all(all_checks)
    result = {
        "schema": "kedf-ti-formal-preflight-v1",
        "status": "verified" if verified else "failed",
        "launch_authorized": verified,
        "checks": top_checks,
        "windows": windows,
    }
    preflight_path = root / "preflight.json"
    preflight_path.write_text(json.dumps(result, indent=2) + "\n")
    checksum_paths.append(preflight_path)
    checksum_lines = [
        f"{sha256(path)}  {path}"
        for path in sorted(checksum_paths, key=lambda item: str(item))
    ]
    (root / "INPUT_SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--target-kedf", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--ranks", type=int, default=12)
    parser.add_argument("--source-step", type=int, default=295)
    parser.add_argument("--minimum-nn", type=float, default=2.0)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--tau-override", action="append", default=[])
    parser.add_argument("--source-step-override", action="append", default=[])
    args = parser.parse_args()
    result = verify(
        args.root,
        target_kedf=args.target_kedf,
        steps=args.steps,
        ranks=args.ranks,
        source_step=args.source_step,
        minimum_nn=args.minimum_nn,
        default_tau=args.csvr_tau,
        tau_overrides=parse_tau_overrides(args.tau_override),
        source_step_overrides=parse_source_step_overrides(
            args.source_step_override
        ),
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
