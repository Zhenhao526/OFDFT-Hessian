#!/usr/bin/env python3
"""Continue selected KEDF-reference TI windows from their own final frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_ti_windows import SUPPORTED_KEDFS


ROOT = Path(__file__).resolve().parents[1]


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
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    phase_roots = resolve_phase_roots(args.parent, list(args.phases), args.phase_roots)

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
        pair_model = Path(parent_manifest["pair_model"]).resolve()
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
        for window_index, window in enumerate(selected_windows):
            source_dir = parent_phase / window["label"]
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
            "thermalized_initial": True,
            "windows": windows,
        }
        phase_out.mkdir(parents=True, exist_ok=True)
        (phase_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        prepared[phase] = [window["label"] for window in windows]

    print(json.dumps({"out": str(out), "prepared": prepared}, indent=2))


if __name__ == "__main__":
    main()
