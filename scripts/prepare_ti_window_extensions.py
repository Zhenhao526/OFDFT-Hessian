#!/usr/bin/env python3
"""Continue selected KEDF-reference TI windows from their own final frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_ti_windows import SUPPORTED_KEDFS


ROOT = Path(__file__).resolve().parents[1]


def resolve_pair_model(
    parent_pair_model: Path,
    override: Path | None,
    *,
    target_kedf: str,
    phase: str,
    selected_windows: list[dict],
) -> tuple[Path, bool]:
    if override is None:
        return parent_pair_model.resolve(), False
    if any(abs(float(window["lambda"]) - 1.0) > 1.0e-12 for window in selected_windows):
        raise ValueError("pair-model override is only valid for lambda=1 windows")
    resolved = override.resolve()
    document = load_json(resolved)
    model_phase = document.get("phase", document.get("reference_phase"))
    if str(document.get("target_kedf", "")).lower() != target_kedf:
        raise ValueError("pair-model override has the wrong target KEDF")
    if model_phase != phase:
        raise ValueError("pair-model override has the wrong phase provenance")
    if document.get("reference_gate_passed") is not True:
        raise ValueError("pair-model override did not pass its reference gate")
    if document.get("short_range_guard_passed") is not True:
        raise ValueError("pair-model override did not pass its short-range guard")
    return resolved, True


def resolve_phase_roots(
    parent: Path | None,
    phases: list[str],
    phase_root_specs: list[str] | None,
) -> dict[str, Path]:
    if phase_root_specs:
        roots: dict[str, Path] = {}
        for spec in phase_root_specs:
            phase, separator, path = spec.partition("=")
            if not separator or phase not in {"solid", "liquid"} or not path:
                raise ValueError(
                    "--phase-root must use solid=PATH or liquid=PATH"
                )
            if phase in roots:
                raise ValueError(f"duplicate --phase-root for {phase}")
            roots[phase] = Path(path).resolve()
        return roots
    if parent is None:
        raise ValueError("provide --parent or at least one --phase-root")
    resolved_parent = parent.resolve()
    return {phase: resolved_parent / phase for phase in phases}


def parse_source_run_overrides(specs: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for spec in specs or []:
        phase, separator, path = spec.partition("=")
        if not separator or phase not in {"solid", "liquid"} or not path:
            raise ValueError(
                "--source-run-override must use solid=PATH or liquid=PATH"
            )
        if phase in overrides:
            raise ValueError(f"duplicate --source-run-override for {phase}")
        overrides[phase] = Path(path).resolve()
    return overrides


def validate_source_run_override(
    override: Path,
    *,
    target_kedf: str,
    phase: str,
    volume_per_atom_A3: float,
    selected_windows: list[dict],
) -> Path:
    if any(abs(float(window["lambda"]) - 1.0) > 1.0e-12 for window in selected_windows):
        raise ValueError("source-run override is only valid for lambda=1 windows")
    resolved = override.resolve()
    metadata = load_json(resolved / "metadata.json")
    phase_analysis = load_json(resolved / "phase_analysis.json")
    if str(metadata.get("target_kedf", "")).lower() != target_kedf:
        raise ValueError("source-run override has the wrong target KEDF")
    if metadata.get("phase") != phase:
        raise ValueError("source-run override has the wrong phase provenance")
    if phase_analysis.get("status") != f"{phase}_verified":
        raise ValueError("source-run override did not pass its phase gate")
    source_volume = float(metadata["volume_per_atom_A3"])
    if not math.isclose(
        source_volume,
        volume_per_atom_A3,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError("source-run override has a different volume")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path)
    parser.add_argument(
        "--phase-root",
        action="append",
        dest="phase_roots",
        help="explicit phase source as solid=PATH or liquid=PATH; may be repeated",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--csvr-tau", type=float, default=10.0)
    parser.add_argument("--dumpfreq", type=int, default=5)
    parser.add_argument("--restartfreq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202608000)
    parser.add_argument("--ranks", type=int, default=12)
    parser.add_argument(
        "--phases", nargs="+", choices=("solid", "liquid"), default=("solid", "liquid")
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="lambda labels to continue; omit to continue every window",
    )
    parser.add_argument(
        "--pair-model-override",
        type=Path,
        help=(
            "replace the parent pair model for selected lambda=1 windows only; "
            "the replacement must have matching KEDF and phase provenance"
        ),
    )
    parser.add_argument(
        "--source-run-override",
        action="append",
        dest="source_run_overrides",
        help=(
            "use a verified same-KEDF, same-phase, same-volume source for "
            "selected lambda=1 windows as solid=PATH or liquid=PATH"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    phase_roots = resolve_phase_roots(args.parent, list(args.phases), args.phase_roots)
    source_run_overrides = parse_source_run_overrides(args.source_run_overrides)
    unknown_override_phases = sorted(set(source_run_overrides) - set(phase_roots))
    if unknown_override_phases:
        raise ValueError(
            "source-run override has no matching phase root: "
            + ", ".join(unknown_override_phases)
        )

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
            "md_csvr_tau": args.csvr_tau,
            "md_dumpfreq": args.dumpfreq,
            "md_restartfreq": args.restartfreq,
            "init_vel": 1,
            "mpirun_np": args.ranks,
        }
    )
    element = load_json(ROOT / "config" / "al.json")

    prepared = {}
    for phase_index, (phase, parent_phase) in enumerate(phase_roots.items()):
        parent_manifest = json.loads((parent_phase / "manifest.json").read_text())
        parent_kedf = str(parent_manifest.get("target_kedf", "")).lower()
        if parent_kedf != target_kedf:
            raise ValueError(
                f"{phase} parent target {parent_kedf!r} does not match "
                f"config target {target_kedf!r}"
            )
        temperature = float(parent_manifest["target_temperature_K"])
        windows = []

        available_labels = {str(window["label"]) for window in parent_manifest["windows"]}
        selected_labels = set(args.labels) if args.labels else available_labels
        unknown = sorted(selected_labels - available_labels)
        if unknown:
            raise ValueError(
                f"requested labels not present for {phase}: {', '.join(unknown)}"
            )
        selected_windows = [
            window
            for window in parent_manifest["windows"]
            if str(window["label"]) in selected_labels
        ]
        pair_model, pair_model_overridden = resolve_pair_model(
            Path(parent_manifest["pair_model"]),
            args.pair_model_override,
            target_kedf=target_kedf,
            phase=phase,
            selected_windows=selected_windows,
        )
        source_run_override = None
        if phase in source_run_overrides:
            source_run_override = validate_source_run_override(
                source_run_overrides[phase],
                target_kedf=target_kedf,
                phase=phase,
                volume_per_atom_A3=float(parent_manifest["volume_per_atom_A3"]),
                selected_windows=selected_windows,
            )
        for window_index, window in enumerate(selected_windows):
            source_dir = source_run_override or parent_phase / window["label"]
            source = load_atom_source(
                source_dir.as_posix(), "last", "Al", include_velocities=True
            )
            atoms = source["atoms"]
            if atoms.velocities is None:
                raise RuntimeError(f"missing source velocities at {source_dir}")

            point_config = dict(config)
            point_config.update(
                {
                    "md_tfirst": temperature,
                    "md_tlast": temperature,
                    "md_seed": args.seed + 1000 * phase_index + window_index,
                }
            )
            write_job(
                out / phase / window["label"],
                atoms,
                element,
                point_config,
                job_type=(
                    f"{target_kedf}_pair_thermodynamic_integration_continuation"
                ),
                suffix=(
                    f"al{atoms.natoms}_{phase}_T{int(temperature):04d}_"
                    f"{window['label']}_continuation"
                ),
                calculation="md",
                extra_metadata={
                    "phase": phase,
                    "target_kedf": target_kedf,
                    "lambda": window["lambda"],
                    "target_temperature_K": temperature,
                    "volume_per_atom_A3": parent_manifest["volume_per_atom_A3"],
                    "volume_A3": parent_manifest["volume_A3"],
                    "source": source["source"],
                    "source_step": source["step"],
                    "source_velocities_discarded": False,
                    "pair_model": str(pair_model),
                    "pair_model_override_at_lambda_one": pair_model_overridden,
                    "source_run_override_at_lambda_one": (
                        source_run_override is not None
                    ),
                    "steps": args.steps,
                    "csvr_tau": args.csvr_tau,
                    "mpi_ranks": args.ranks,
                },
            )
            windows.append(
                {
                    "label": window["label"],
                    "lambda": window["lambda"],
                    "seed": point_config["md_seed"],
                }
            )

        phase_out = out / phase
        manifest = {
            "schema": (
                "wt-pair-ti-window-continuations-v1"
                if target_kedf == "wt"
                else "kedf-pair-ti-window-continuations-v1"
            ),
            "phase": phase,
            "target_kedf": target_kedf,
            "target_temperature_K": temperature,
            "volume_per_atom_A3": parent_manifest["volume_per_atom_A3"],
            "volume_A3": parent_manifest["volume_A3"],
            "natoms": parent_manifest["natoms"],
            "steps": args.steps,
            "mpi_ranks": args.ranks,
            "parent": str(parent_phase),
            "pair_model": str(pair_model),
            "pair_model_override_at_lambda_one": pair_model_overridden,
            "source_run_override_at_lambda_one": source_run_override is not None,
            "source_run_override": (
                str(source_run_override) if source_run_override is not None else None
            ),
            "thermalized_initial": True,
            "windows": windows,
        }
        phase_out.mkdir(parents=True, exist_ok=True)
        (phase_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        prepared[phase] = [window["label"] for window in windows]

    print(json.dumps({"out": str(out), "prepared": prepared}, indent=2))


if __name__ == "__main__":
    main()
