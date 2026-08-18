#!/usr/bin/env python3
"""Prepare and analyze OFDFT-KEDF-to-pair thermodynamic-integration windows."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.free_energy import exponential_free_energy_difference
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import lattice_volume
from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_mg_phase_md import hcp_phase_checks
from scripts.analyze_two_phase_run import parse_md_log, series_stats

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_KEDFS = {"wt", "xwm", "lkt", "ext-wt"}
COMPONENT_RE = re.compile(
    r"MPN_TI_COMPONENTS step=(\d+) lambda=([-+0-9.eE]+) "
    r"U_MPN_eV=([-+0-9.eE]+) U_REF_eV=([-+0-9.eE]+) "
    r"(?:W_REF_eV=[-+0-9.eE]+ )?"
    r"DELTA_U_eV=([-+0-9.eE]+) RMIN_A=([-+0-9.eE]+)"
)


def scaled_to_volume(atoms: AtomSet, target_volume_A3: float) -> AtomSet:
    current = lattice_volume(tuple(atoms.lattice_vectors))
    factor = (target_volume_A3 / current) ** (1.0 / 3.0)
    return AtomSet(
        list(atoms.symbols),
        list(atoms.scaled_positions),
        [tuple(value * factor for value in vector) for vector in atoms.lattice_vectors],
        velocities=list(atoms.velocities) if atoms.velocities is not None else None,
        movements=list(atoms.movements) if atoms.movements is not None else None,
    )


def lambda_label(value: float) -> str:
    return f"lambda_{value:.3f}".replace(".", "p")


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    element_symbol = str(getattr(args, "element_symbol", "Al"))
    element_config = Path(
        getattr(args, "element_config", ROOT / "config" / "al.json")
    )
    source = load_atom_source(
        args.source,
        args.source_frame,
        element_symbol,
        include_velocities=False,
    )
    atoms = source["atoms"]
    atoms = scaled_to_volume(atoms, args.volume_per_atom * atoms.natoms)
    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    if target_kedf not in SUPPORTED_KEDFS:
        raise ValueError(
            f"free-energy windows require one of {sorted(SUPPORTED_KEDFS)}, "
            f"got {target_kedf!r}"
        )
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 0,
            "md_type": "nvt",
            "md_nstep": args.steps,
            "md_dt": args.dt,
            "md_tfirst": args.temperature,
            "md_tlast": args.temperature,
            "md_thermostat": "csvr",
            "md_csvr_tau": args.csvr_tau,
            "md_dumpfreq": args.dumpfreq,
            "md_restartfreq": args.restartfreq,
            "init_vel": 0,
        }
    )
    ranks = getattr(args, "ranks", None)
    if ranks is not None:
        config["mpirun_np"] = ranks
    element = load_json(element_config)
    if str(element.get("element")) != element_symbol:
        raise ValueError(
            f"element config declares {element.get('element')!r}, "
            f"expected {element_symbol!r}"
        )
    windows = []
    for index, lambda_value in enumerate(sorted(set(args.lambdas))):
        if not 0.0 <= lambda_value <= 1.0:
            raise ValueError("all lambda values must be in [0, 1]")
        label = lambda_label(lambda_value)
        point_config = dict(config)
        point_config["md_seed"] = args.seed + index
        write_job(
            out / label,
            atoms,
            element,
            point_config,
            job_type=f"{target_kedf}_pair_thermodynamic_integration",
            suffix=(
                f"{element_symbol.lower()}{atoms.natoms}_{target_kedf}_{args.phase}_"
                f"T{int(args.temperature):04d}_{label}"
            ),
            calculation="md",
            extra_metadata={
                "phase": args.phase,
                "element": element_symbol,
                "target_kedf": target_kedf,
                "lambda": lambda_value,
                "target_temperature_K": args.temperature,
                "volume_per_atom_A3": args.volume_per_atom,
                "volume_A3": args.volume_per_atom * atoms.natoms,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": True,
                "pair_model": str(args.pair_model.resolve()),
                "steps": args.steps,
                "csvr_tau": args.csvr_tau,
            },
        )
        windows.append({"label": label, "lambda": lambda_value, "seed": args.seed + index})
    manifest = {
        "schema": (
            "wt-pair-ti-windows-v2"
            if target_kedf == "wt"
            else "kedf-pair-ti-windows-v1"
        ),
        "phase": args.phase,
        "element": element_symbol,
        "element_config": str(element_config.resolve()),
        "target_kedf": target_kedf,
        "target_temperature_K": args.temperature,
        "volume_per_atom_A3": args.volume_per_atom,
        "volume_A3": args.volume_per_atom * atoms.natoms,
        "natoms": atoms.natoms,
        "steps": args.steps,
        "source": source["source"],
        "source_step": source["step"],
        "pair_model": str(args.pair_model.resolve()),
        "windows": windows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_components(path: Path) -> list[dict]:
    rows = []
    for match in COMPONENT_RE.finditer(path.read_text(errors="replace")):
        step, lambda_value, mpn, reference, delta, nearest = match.groups()
        rows.append(
            {
                "step": int(step),
                "lambda": float(lambda_value),
                "U_MPN_eV": float(mpn),
                "U_target_eV": float(mpn),
                "U_REF_eV": float(reference),
                "delta_U_eV": float(delta),
                "nearest_neighbor_A": float(nearest),
            }
        )
    return rows


def trapezoid(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    ordered = sorted(rows, key=lambda row: row["lambda"])
    return sum(
        0.5
        * (right["lambda"] - left["lambda"])
        * (left["delta_U_last_half_eV"]["mean"] + right["delta_U_last_half_eV"]["mean"])
        for left, right in zip(ordered, ordered[1:])
    )


def apply_element_phase_model(phase: dict, manifest: dict) -> dict:
    if str(manifest.get("element")) != "Mg":
        return phase
    checks = hcp_phase_checks(
        phase,
        str(manifest["phase"]),
        thermalized_initial=True,
    )
    expected = str(manifest["phase"])
    phase["legacy_structure_status"] = phase.get("status")
    phase["status"] = (
        f"{expected}_verified" if all(checks.values()) else f"{expected}_not_verified"
    )
    phase["structure_model"] = "hcp_diffusion_and_non_affine_msd_v2"
    phase["hcp_phase_gate"] = checks
    return phase


def analyze(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    raw_by_lambda = {}
    for window in manifest["windows"]:
        run = root / window["label"]
        logs = sorted(run.glob("OUT.*/running_md.log"))
        if not logs:
            rows.append({**window, "status": "missing_output"})
            continue
        components = parse_components(logs[-1])
        md_rows, max_step = parse_md_log(logs[-1])
        late_components = components[len(components) // 2 :]
        late_md = md_rows[len(md_rows) // 2 :]
        # TI windows inherit positions from a phase-verified thermalized source.
        # Regenerating velocities does not turn that structure into a pristine
        # fresh-preparation sample, so validate it as a continuation.
        phase = apply_element_phase_model(
            analyze_phase(run, manifest["phase"], thermalized_initial=True),
            manifest,
        )
        (run / "phase_analysis.json").write_text(json.dumps(phase, indent=2, sort_keys=True) + "\n")
        delta_values = [row["delta_U_eV"] for row in late_components]
        raw_by_lambda[window["lambda"]] = delta_values
        rows.append(
            {
                **window,
                "max_step": max_step,
                "component_samples": len(components),
                "phase_status": phase["status"],
                "temperature_last_half_K": series_stats([row["temperature_K"] for row in late_md]),
                "delta_U_last_half_eV": series_stats(delta_values),
                "delta_U_last_half_meV_per_atom": {
                    "mean": statistics.mean(delta_values) * 1000.0 / manifest["natoms"],
                    "sd": statistics.stdev(delta_values) * 1000.0 / manifest["natoms"]
                    if len(delta_values) > 1
                    else 0.0,
                }
                if delta_values
                else {},
                "minimum_pair_distance_A": min((row["nearest_neighbor_A"] for row in components), default=None),
            }
        )
    complete = [
        row
        for row in rows
        if row.get("max_step", -1) >= manifest["steps"]
        and row.get("phase_status") == f"{manifest['phase']}_verified"
        and row.get("delta_U_last_half_eV")
    ]
    overlaps = []
    temperature = manifest["target_temperature_K"]
    for left, right in zip(sorted(complete, key=lambda row: row["lambda"]), sorted(complete, key=lambda row: row["lambda"])[1:]):
        delta_lambda = right["lambda"] - left["lambda"]
        forward = exponential_free_energy_difference(
            [delta_lambda * value for value in raw_by_lambda[left["lambda"]]], temperature
        )
        reverse = exponential_free_energy_difference(
            [-delta_lambda * value for value in raw_by_lambda[right["lambda"]]], temperature
        )
        overlaps.append(
            {
                "lambda_left": left["lambda"],
                "lambda_right": right["lambda"],
                "forward_effective_sample_fraction": forward["effective_sample_fraction"],
                "reverse_effective_sample_fraction": reverse["effective_sample_fraction"],
            }
        )
    delta_f = trapezoid(complete)
    result = {
        **manifest,
        "window_results": rows,
        "complete_phase_valid_windows": len(complete),
        "delta_F_target_minus_reference_eV": delta_f,
        "delta_F_target_minus_reference_meV_per_atom": delta_f
        * 1000.0
        / manifest["natoms"]
        if delta_f is not None
        else None,
        "adjacent_overlap": overlaps,
    }
    if str(manifest.get("target_kedf", "")).lower() == "wt":
        result.update(
            {
                "delta_F_MPN_minus_reference_eV": delta_f,
                "delta_F_MPN_minus_reference_meV_per_atom": delta_f
                * 1000.0
                / manifest["natoms"]
                if delta_f is not None
                else None,
            }
        )
    (root / "ti_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--source-frame", default="last")
    prep.add_argument("--phase", choices=("solid", "liquid"), required=True)
    prep.add_argument("--temperature", type=float, required=True)
    prep.add_argument("--volume-per-atom", type=float, required=True)
    prep.add_argument("--lambdas", type=float, nargs="+", required=True)
    prep.add_argument("--steps", type=int, default=500)
    prep.add_argument("--dt", type=float, default=1.0)
    prep.add_argument("--csvr-tau", type=float, default=20.0)
    prep.add_argument("--dumpfreq", type=int, default=5)
    prep.add_argument("--restartfreq", type=int, default=100)
    prep.add_argument("--seed", type=int, default=20260720)
    prep.add_argument("--pair-model", type=Path, required=True)
    prep.add_argument("--ranks", type=int)
    prep.add_argument("--element-symbol", default="Al")
    prep.add_argument("--element-config", type=Path, default=ROOT / "config" / "al.json")
    prep.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    prep.set_defaults(func=prepare)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("run_root", type=Path)
    analysis.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
