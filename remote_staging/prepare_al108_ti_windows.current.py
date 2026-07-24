#!/usr/bin/env python3
"""Prepare and analyze MPN-to-pair thermodynamic-integration windows."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.free_energy import exponential_free_energy_difference
from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats
from scripts.prepare_al108_volume_scan import scaled_to_volume

ROOT = Path(__file__).resolve().parents[1]
COMPONENT_RE = re.compile(
    r"MPN_TI_COMPONENTS step=(\d+) lambda=([-+0-9.eE]+) "
    r"U_MPN_eV=([-+0-9.eE]+) U_REF_eV=([-+0-9.eE]+) "
    r"DELTA_U_eV=([-+0-9.eE]+) RMIN_A=([-+0-9.eE]+)"
)


def lambda_label(value: float) -> str:
    return f"lambda_{value:.3f}".replace(".", "p")


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source_by_lambda = {}
    for item in args.source_map:
        lambda_text, source_path = item.split("=", 1)
        source_by_lambda[float(lambda_text)] = source_path
    if set(source_by_lambda) - set(args.lambdas):
        raise ValueError("--source-map contains lambda values not requested by --lambdas")
    config = load_json(args.config)
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
            "mpirun_np": 12,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    windows = []
    natoms = None
    for index, lambda_value in enumerate(sorted(set(args.lambdas))):
        if not 0.0 <= lambda_value <= 1.0:
            raise ValueError("all lambda values must be in [0, 1]")
        source_path = source_by_lambda.get(lambda_value, args.source)
        source = load_atom_source(
            source_path,
            args.source_frame,
            "Al",
            include_velocities=args.preserve_velocities,
        )
        atoms = source["atoms"]
        if args.preserve_velocities and atoms.velocities is None:
            raise ValueError(f"source for lambda={lambda_value} does not contain velocities")
        atoms = scaled_to_volume(atoms, args.volume_per_atom * atoms.natoms)
        if natoms is not None and atoms.natoms != natoms:
            raise ValueError("all lambda sources must contain the same number of atoms")
        natoms = atoms.natoms
        label = lambda_label(lambda_value)
        point_config = dict(config)
        point_config["md_seed"] = args.seed + index
        write_job(
            out / label,
            atoms,
            element,
            point_config,
            job_type="mpn_pair_thermodynamic_integration",
            suffix=f"al108_{args.phase}_T{int(args.temperature):04d}_{label}",
            calculation="md",
            extra_metadata={
                "phase": args.phase,
                "lambda": lambda_value,
                "target_temperature_K": args.temperature,
                "volume_per_atom_A3": args.volume_per_atom,
                "volume_A3": args.volume_per_atom * atoms.natoms,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": not args.preserve_velocities,
                "pair_model": str(args.pair_model.resolve()),
                "steps": args.steps,
                "csvr_tau": args.csvr_tau,
            },
        )
        windows.append(
            {
                "label": label,
                "lambda": lambda_value,
                "seed": args.seed + index,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": not args.preserve_velocities,
            }
        )
    if natoms is None:
        raise ValueError("no lambda windows requested")
    manifest = {
        "schema": "mpn-pair-ti-windows-v1",
        "phase": args.phase,
        "target_temperature_K": args.temperature,
        "volume_per_atom_A3": args.volume_per_atom,
        "volume_A3": args.volume_per_atom * natoms,
        "natoms": natoms,
        "steps": args.steps,
        "source": args.source,
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


def simpson(rows: list[dict]) -> float | None:
    """Composite Simpson integral for an odd, uniformly spaced lambda grid."""
    ordered = sorted(rows, key=lambda row: row["lambda"])
    if len(ordered) < 3 or len(ordered) % 2 == 0:
        return None
    spacings = [
        right["lambda"] - left["lambda"]
        for left, right in zip(ordered, ordered[1:])
    ]
    step = spacings[0]
    if step <= 0.0 or any(not math.isclose(value, step, rel_tol=1e-9, abs_tol=1e-12) for value in spacings):
        return None
    values = [row["delta_U_last_half_eV"]["mean"] for row in ordered]
    return step / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * sum(values[1:-1:2])
        + 2.0 * sum(values[2:-1:2])
    )


def _fermi(value: float) -> float:
    if value > 50.0:
        return math.exp(-value)
    if value < -50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(value))


def bennett_acceptance_ratio(
    left_delta_u: list[float],
    right_delta_u: list[float],
    delta_lambda: float,
    temperature_K: float,
) -> float:
    """BAR free-energy difference between two equally sampled lambda states."""
    if not left_delta_u or not right_delta_u:
        raise ValueError("BAR requires samples from both lambda states")
    if temperature_K <= 0.0 or delta_lambda <= 0.0:
        raise ValueError("temperature and delta lambda must be positive")
    beta = 1.0 / (8.617333262145e-5 * temperature_K)
    forward = [delta_lambda * value for value in left_delta_u]
    reverse = [-delta_lambda * value for value in right_delta_u]
    sample_ratio = math.log(len(forward) / len(reverse))

    def residual(delta_f: float) -> float:
        left = sum(_fermi(beta * (work - delta_f) + sample_ratio) for work in forward)
        right = sum(_fermi(beta * (work + delta_f) - sample_ratio) for work in reverse)
        return left - right

    lower = min(forward + [-value for value in reverse]) - 1.0
    upper = max(forward + [-value for value in reverse]) + 1.0
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def block_bootstrap_integrals(
    rows: list[dict],
    raw_by_lambda: dict[float, list[float]],
    natoms: int,
    samples: int = 4000,
    block_size: int = 10,
    seed: int = 20260720,
) -> dict:
    """Estimate correlated-sampling uncertainty for trapezoid and Simpson TI."""
    ordered = sorted(rows, key=lambda row: row["lambda"])
    blocks_by_lambda = {}
    for row in ordered:
        values = raw_by_lambda[row["lambda"]]
        blocks = [
            values[index : index + block_size]
            for index in range(0, len(values), block_size)
            if len(values[index : index + block_size]) == block_size
        ]
        if len(blocks) < 2:
            return {"status": "insufficient_blocks", "block_size": block_size}
        blocks_by_lambda[row["lambda"]] = blocks

    rng = random.Random(seed)
    trapezoid_samples = []
    simpson_samples = []
    for _ in range(samples):
        replica_rows = []
        for row in ordered:
            blocks = blocks_by_lambda[row["lambda"]]
            replica = [
                value
                for _ in range(len(blocks))
                for value in rng.choice(blocks)
            ]
            replica_rows.append(
                {
                    "lambda": row["lambda"],
                    "delta_U_last_half_eV": {"mean": statistics.mean(replica)},
                }
            )
        trapezoid_samples.append(trapezoid(replica_rows))
        simpson_value = simpson(replica_rows)
        if simpson_value is not None:
            simpson_samples.append(simpson_value)

    def summarize(values: list[float]) -> dict:
        converted = sorted(value * 1000.0 / natoms for value in values)
        return {
            "mean_meV_per_atom": statistics.mean(converted),
            "sd_meV_per_atom": statistics.stdev(converted),
            "ci95_meV_per_atom": [
                _percentile(converted, 0.025),
                _percentile(converted, 0.975),
            ],
        }

    result = {
        "status": "ok",
        "samples": samples,
        "block_size_component_samples": block_size,
        "trapezoid": summarize(trapezoid_samples),
    }
    if simpson_samples:
        result["simpson"] = summarize(simpson_samples)
    return result


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
        phase = analyze_phase(
            run,
            manifest["phase"],
            thermalized_initial=not window.get("source_velocities_discarded", True),
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
    bar_segments = []
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
        bar_segments.append(
            bennett_acceptance_ratio(
                raw_by_lambda[left["lambda"]],
                raw_by_lambda[right["lambda"]],
                delta_lambda,
                temperature,
            )
        )
    delta_f = trapezoid(complete)
    simpson_delta_f = simpson(complete)
    bar_delta_f = sum(bar_segments) if bar_segments else None
    bootstrap = (
        block_bootstrap_integrals(complete, raw_by_lambda, manifest["natoms"])
        if len(complete) == len(manifest["windows"])
        else {"status": "incomplete_windows"}
    )
    result = {
        **manifest,
        "window_results": rows,
        "complete_phase_valid_windows": len(complete),
        "delta_F_MPN_minus_reference_eV": delta_f,
        "delta_F_MPN_minus_reference_meV_per_atom": delta_f * 1000.0 / manifest["natoms"]
        if delta_f is not None
        else None,
        "delta_F_simpson_MPN_minus_reference_eV": simpson_delta_f,
        "delta_F_simpson_MPN_minus_reference_meV_per_atom": simpson_delta_f
        * 1000.0
        / manifest["natoms"]
        if simpson_delta_f is not None
        else None,
        "delta_F_BAR_MPN_minus_reference_eV": bar_delta_f,
        "delta_F_BAR_MPN_minus_reference_meV_per_atom": bar_delta_f
        * 1000.0
        / manifest["natoms"]
        if bar_delta_f is not None
        else None,
        "block_bootstrap": bootstrap,
        "adjacent_overlap": overlaps,
    }
    (root / "ti_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument(
        "--source-map",
        action="append",
        default=[],
        metavar="LAMBDA=PATH",
        help="override --source for a specific lambda; may be repeated",
    )
    prep.add_argument("--source-frame", default="last")
    prep.add_argument("--preserve-velocities", action="store_true")
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
    prep.add_argument("--config", type=Path, default=ROOT / "config" / "abacus_mpn_node04_cpu12_stress.json")
    prep.set_defaults(func=prepare)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("run_root", type=Path)
    analysis.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
