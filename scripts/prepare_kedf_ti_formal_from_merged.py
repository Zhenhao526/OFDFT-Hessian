#!/usr/bin/env python3
"""Prepare formal KEDF-to-pair TI windows from a verified merged pilot grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_ti_windows import SUPPORTED_KEDFS


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tau_overrides(values: list[str]) -> dict[tuple[str, str], float]:
    overrides: dict[tuple[str, str], float] = {}
    for value in values:
        key, separator, raw_tau = value.partition("=")
        phase, phase_separator, label = key.partition(":")
        if (
            not separator
            or not phase_separator
            or phase not in {"solid", "liquid"}
            or not label
            or not raw_tau
        ):
            raise ValueError(
                "--tau-override must use solid:lambda_LABEL=TAU or "
                "liquid:lambda_LABEL=TAU"
            )
        tau = float(raw_tau)
        if tau <= 0.0:
            raise ValueError("CSVR tau must be positive")
        key_tuple = (phase, label)
        if key_tuple in overrides:
            raise ValueError(f"duplicate tau override for {phase}:{label}")
        overrides[key_tuple] = tau
    return overrides


def selected_rows(merged: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    phase_result = merged["phase_results"][phase]
    if phase_result.get("status") != "verified":
        raise ValueError(f"{phase} merged pilot is not verified")
    rows = sorted(phase_result["window_results"], key=lambda row: row["lambda"])
    if len(rows) != 9 or any(row.get("status") != "verified" for row in rows):
        raise ValueError(f"{phase} merged pilot must contain nine verified windows")
    return rows


def final_structure_path(source: dict[str, Any]) -> Path:
    source_path = Path(source["source"])
    if source_path.is_file():
        return source_path
    candidate = source_path / f"STRU_MD_{int(source['step'])}"
    if not candidate.is_file():
        raise FileNotFoundError(f"missing final structure {candidate}")
    return candidate


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    merged_path = args.merged.resolve()
    merged = json.loads(merged_path.read_text())
    if merged.get("status") != "pilot_grid_verified":
        raise ValueError("merged pilot grid is not verified")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    if target_kedf not in SUPPORTED_KEDFS:
        raise ValueError(f"unsupported KEDF {target_kedf!r}")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 0,
            "md_type": "nvt",
            "md_nstep": args.steps,
            "md_dt": args.dt,
            "md_thermostat": "csvr",
            "md_dumpfreq": args.dumpfreq,
            "md_restartfreq": args.restartfreq,
            "init_vel": 1,
            "mpirun_np": args.ranks,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    tau_overrides = parse_tau_overrides(args.tau_override)
    known_keys: set[tuple[str, str]] = set()
    pair_models: set[Path] = set()
    phase_manifests = {}
    source_records = []

    for phase_index, phase in enumerate(("solid", "liquid")):
        rows = selected_rows(merged, phase)
        windows = []
        phase_metadata = []
        for window_index, row in enumerate(rows):
            label = str(row["label"])
            key = (phase, label)
            known_keys.add(key)
            source_run = Path(row["run"]).resolve()
            source_analysis = json.loads(
                (source_run / "phase_analysis.json").read_text()
            )
            if source_analysis.get("status") != f"{phase}_verified":
                raise ValueError(f"{phase}:{label} source phase is not verified")
            source_metadata = json.loads(
                (source_run / "metadata.json").read_text()
            )
            if str(source_metadata.get("target_kedf", "")).lower() != target_kedf:
                raise ValueError(f"{phase}:{label} source KEDF mismatch")
            pair_model = Path(source_metadata["pair_model"]).resolve()
            if not pair_model.is_file():
                raise FileNotFoundError(pair_model)
            pair_models.add(pair_model)

            source = load_atom_source(
                source_run.as_posix(), "last", "Al", include_velocities=True
            )
            atoms = source["atoms"]
            if atoms.velocities is None:
                raise RuntimeError(f"missing source velocities at {source_run}")
            source_structure = final_structure_path(source)
            tau = tau_overrides.get(key, args.csvr_tau)
            point_config = dict(config)
            point_config.update(
                {
                    "md_csvr_tau": tau,
                    "md_tfirst": float(source_metadata["target_temperature_K"]),
                    "md_tlast": float(source_metadata["target_temperature_K"]),
                    "md_seed": args.seed + 1000 * phase_index + window_index,
                }
            )
            write_job(
                output / phase / label,
                atoms,
                element,
                point_config,
                job_type=f"{target_kedf}_pair_formal_thermodynamic_integration",
                suffix=(
                    f"al{atoms.natoms}_{phase}_T"
                    f"{int(source_metadata['target_temperature_K']):04d}_"
                    f"{label}_formal"
                ),
                calculation="md",
                extra_metadata={
                    "phase": phase,
                    "target_kedf": target_kedf,
                    "lambda": row["lambda"],
                    "target_temperature_K": source_metadata[
                        "target_temperature_K"
                    ],
                    "volume_per_atom_A3": source_metadata[
                        "volume_per_atom_A3"
                    ],
                    "volume_A3": source_metadata["volume_A3"],
                    "source_run": str(source_run),
                    "source": source["source"],
                    "source_step": source["step"],
                    "source_structure": str(source_structure),
                    "source_structure_sha256": sha256(source_structure),
                    "source_velocities_discarded": False,
                    "pair_model": str(pair_model),
                    "pair_model_sha256": sha256(pair_model),
                    "merged_pilot_analysis": str(merged_path),
                    "merged_pilot_analysis_sha256": sha256(merged_path),
                    "steps": args.steps,
                    "csvr_tau": tau,
                    "mpi_ranks": args.ranks,
                },
            )
            windows.append(
                {
                    "label": label,
                    "lambda": row["lambda"],
                    "seed": point_config["md_seed"],
                    "csvr_tau": tau,
                    "source_run": str(source_run),
                }
            )
            source_record = {
                "phase": phase,
                "label": label,
                "lambda": row["lambda"],
                "source_run": str(source_run),
                "source_step": source["step"],
                "source_structure": str(source_structure),
                "source_structure_sha256": sha256(source_structure),
                "source_phase_status": source_analysis["status"],
                "source_minimum_nearest_neighbor_A": source_analysis[
                    "trajectory"
                ]["minimum_nearest_neighbor_A"],
                "csvr_tau": tau,
            }
            phase_metadata.append(source_record)
            source_records.append(source_record)

        first_metadata = json.loads(
            (Path(rows[0]["run"]) / "metadata.json").read_text()
        )
        pair_model = Path(first_metadata["pair_model"]).resolve()
        manifest = {
            "schema": "kedf-pair-ti-formal-production-v1",
            "phase": phase,
            "target_kedf": target_kedf,
            "target_temperature_K": first_metadata["target_temperature_K"],
            "volume_per_atom_A3": first_metadata["volume_per_atom_A3"],
            "volume_A3": first_metadata["volume_A3"],
            "natoms": first_metadata["natoms"],
            "steps": args.steps,
            "mpi_ranks": args.ranks,
            "pair_model": str(pair_model),
            "pair_model_sha256": sha256(pair_model),
            "merged_pilot_analysis": str(merged_path),
            "thermalized_initial": True,
            "windows": windows,
            "source_provenance": phase_metadata,
        }
        phase_out = output / phase
        phase_out.mkdir(parents=True, exist_ok=True)
        (phase_out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        phase_manifests[phase] = str(phase_out / "manifest.json")

    unknown_overrides = sorted(set(tau_overrides) - known_keys)
    if unknown_overrides:
        rendered = ", ".join(
            f"{phase}:{label}" for phase, label in unknown_overrides
        )
        raise ValueError(f"tau overrides not present in merged pilot: {rendered}")
    if len(pair_models) != 1:
        raise ValueError("solid and liquid formal windows use different pair models")
    pair_model = pair_models.pop()
    result = {
        "schema": "kedf-ti-formal-production-manifest-v1",
        "status": "prepared",
        "launch_authorized": False,
        "target_kedf": target_kedf,
        "merged_pilot_analysis": str(merged_path),
        "merged_pilot_analysis_sha256": sha256(merged_path),
        "pair_model": str(pair_model),
        "pair_model_sha256": sha256(pair_model),
        "steps_per_window": args.steps,
        "mpi_ranks_per_window": args.ranks,
        "phase_manifests": phase_manifests,
        "sources": source_records,
    }
    (output / "formal_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--dumpfreq", type=int, default=5)
    parser.add_argument("--restartfreq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202610000)
    parser.add_argument("--ranks", type=int, default=12)
    parser.add_argument("--tau-override", action="append", default=[])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
