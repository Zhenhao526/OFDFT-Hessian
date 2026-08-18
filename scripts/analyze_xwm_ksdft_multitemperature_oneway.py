#!/usr/bin/env python3
"""Analyze the Mg XWM-to-KSDFT multitemperature one-way diagnostic panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

from mpn_melting.gibbs_helmholtz import gibbs_over_temperature, solve_melting_temperature


KB_EV_PER_K = 8.617333262145e-5
NATOMS = 128
ENERGY_RE = re.compile(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")
ENTROPY_RE = re.compile(r"E_entropy\(-TS\)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
ITER_RE = re.compile(r"#ELEC ITER#\s+(\d+)")
FATAL_RE = re.compile(r"SCF IS NOT CONVERGED|\b(?:nan|NaN|FATAL|ERROR)\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(job: Path) -> dict[str, Any]:
    manifest = job / "OUTPUT_SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"missing {manifest}")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        path = job / relative.strip()
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"output SHA mismatch: {path}")
        checked += 1
    return {"sha256": sha256(manifest), "checked_files": checked}


def highest_band_occupation(path: Path) -> tuple[int, float]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0].isdigit():
            rows.append((int(fields[0]), float(fields[2])))
    if not rows:
        raise ValueError(f"no occupation rows in {path}")
    highest = max(index for index, _ in rows)
    return highest, max(value for index, value in rows if index == highest)


def read_job(job: Path) -> dict[str, Any]:
    if not (job / "sp.done").is_file() or (job / "sp.failed").exists():
        raise ValueError(f"job is not complete: {job}")
    if (job / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"nonzero exit code: {job}")
    output_manifest = verify_manifest(job)
    logs = sorted(job.glob("OUT.*/running_scf.log"))
    occupations = sorted(job.glob("OUT.*/eig_occ.txt"))
    if len(logs) != 1 or len(occupations) != 1:
        raise ValueError(f"expected one KS log and occupation file in {job}")
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    energies = [float(value) for value in ENERGY_RE.findall(text)]
    entropy = [(float(ry), float(ev)) for ry, ev in ENTROPY_RE.findall(text)]
    iterations = [int(value) for value in ITER_RE.findall(text)]
    if len(energies) != 1 or len(entropy) != 1 or not iterations:
        raise ValueError(f"non-unique or missing KS fields in {logs[0]}")
    if FATAL_RE.search(text):
        raise ValueError(f"fatal marker in {logs[0]}")
    highest_band, highest_occupation = highest_band_occupation(occupations[0])
    metadata = load(job / "metadata.json")
    mermin = energies[0]
    minus_ts = entropy[0][1]
    u_ks = mermin - minus_ts
    u_xwm = float(metadata["xwm_potential_energy_eV"])
    return {
        "job_id": metadata["job_id"],
        "relative_path": str(job),
        "temperature_K": int(metadata["target_temperature_K"]),
        "phase": metadata["phase"],
        "source_step": int(metadata["source_md_step"]),
        "source_instantaneous_temperature_K": float(
            metadata["source_instantaneous_temperature_K"]
        ),
        "nearest_neighbor_A": float(metadata["nearest_neighbor_A"]),
        "electronic_iterations": iterations[-1],
        "highest_band_index": highest_band,
        "maximum_highest_band_occupation": highest_occupation,
        "final_etot_is_mermin_eV": mermin,
        "electronic_entropy_minus_ts_eV": minus_ts,
        "u_ks_eV": u_ks,
        "u_xwm_eV": u_xwm,
        "delta_mermin_ks_minus_xwm_eV": mermin - u_xwm,
        "delta_internal_ks_minus_xwm_eV": u_ks - u_xwm,
        "metadata_sha256": sha256(job / "metadata.json"),
        "running_log_sha256": sha256(logs[0]),
        "output_manifest": output_manifest,
    }


def fep(values: list[float], temperature: float) -> tuple[float, float, float]:
    beta = 1.0 / (KB_EV_PER_K * temperature)
    exponents = [-beta * value for value in values]
    maximum = max(exponents)
    weights = [math.exp(value - maximum) for value in exponents]
    total = sum(weights)
    log_mean = maximum + math.log(total / len(weights))
    normalized = [value / total for value in weights]
    ess = 1.0 / sum(value * value for value in normalized)
    return -log_mean / beta, ess, ess / len(values)


def mean_stats(values: list[float]) -> dict[str, float]:
    first = statistics.fmean(values[: len(values) // 2])
    second = statistics.fmean(values[len(values) // 2 :])
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "standard_error": statistics.stdev(values) / math.sqrt(len(values)),
        "half_drift": second - first,
    }


def piecewise_root(points: list[tuple[float, float]]) -> float | None:
    ordered = sorted(points)
    for (left_t, left_g), (right_t, right_g) in zip(ordered, ordered[1:]):
        if left_g == 0.0:
            return left_t
        if left_g * right_g <= 0.0:
            return left_t - left_g * (right_t - left_t) / (right_g - left_g)
    return ordered[-1][0] if ordered[-1][1] == 0.0 else None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def bootstrap_summary(values: list[float], requested: int) -> dict[str, Any]:
    if not values:
        return {"requested_samples": requested, "bracketed_samples": 0}
    return {
        "requested_samples": requested,
        "bracketed_samples": len(values),
        "bracketed_fraction": len(values) / requested,
        "mean_K": statistics.fmean(values),
        "median_K": percentile(values, 0.5),
        "ci95_K": [percentile(values, 0.025), percentile(values, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--xwm-anchor", type=Path, required=True)
    parser.add_argument("--enthalpy-summary", type=Path, required=True)
    parser.add_argument("--waiver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = load(root / "PREPARATION_MANIFEST.json")
    xwm_anchor = load(args.xwm_anchor)
    enthalpy_summary = load(args.enthalpy_summary)
    waiver = load(args.waiver)
    rows = [read_job(root / entry["relative_path"]) for entry in manifest["jobs"]]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["temperature_K"], row["phase"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["source_step"])

    validation_checks = {
        "sixty_unique_jobs": len(rows) == 60 and len({row["job_id"] for row in rows}) == 60,
        "six_groups_of_ten": len(grouped) == 6 and all(len(values) == 10 for values in grouped.values()),
        "all_iterations_le_150": max(row["electronic_iterations"] for row in rows) <= 150,
        "all_highest_band_indices_eq_192": all(row["highest_band_index"] == 192 for row in rows),
        "all_highest_band_occupations_le_1e-8": max(
            row["maximum_highest_band_occupation"] for row in rows
        )
        <= 1.0e-8,
        "all_nearest_neighbors_gt_2_A": min(row["nearest_neighbor_A"] for row in rows) > 2.0,
        "all_output_manifests_verified": True,
    }
    if not all(validation_checks.values()):
        failed = [name for name, passed in validation_checks.items() if not passed]
        raise RuntimeError(f"computational validation failed: {failed}")

    temperature_results: dict[str, Any] = {}
    corrected_g_points = []
    corrected_h_points = []
    xwm_g = {
        int(row["temperature_k"]): float(row["delta_g_ev_per_atom"])
        for row in xwm_anchor["melting_results"]["d50"]["delta_g_grid"]
    }
    xwm_h = {
        int(row["temperature_k"]): float(row["liquid_minus_solid"])
        for row in xwm_anchor["melting_results"]["d50"]["enthalpy_points_ev_per_atom"]
    }
    for temperature in (950, 1000, 1050):
        phase_results = {}
        for phase in ("solid", "liquid"):
            phase_rows = grouped[(temperature, phase)]
            delta_m = [row["delta_mermin_ks_minus_xwm_eV"] for row in phase_rows]
            delta_u_per_atom = [
                row["delta_internal_ks_minus_xwm_eV"] / NATOMS for row in phase_rows
            ]
            fep_system, ess, ess_fraction = fep(delta_m, temperature)
            phase_results[phase] = {
                "count": len(phase_rows),
                "fep_delta_mermin_eV_per_atom": fep_system / NATOMS,
                "fep_delta_mermin_meV_per_atom": fep_system * 1000.0 / NATOMS,
                "fep_effective_sample_size": ess,
                "fep_effective_sample_size_fraction": ess_fraction,
                "delta_internal_ks_minus_xwm": mean_stats(delta_u_per_atom),
                "snapshots": phase_rows,
            }
        delta_delta_g = (
            phase_results["liquid"]["fep_delta_mermin_eV_per_atom"]
            - phase_results["solid"]["fep_delta_mermin_eV_per_atom"]
        )
        delta_delta_h = (
            phase_results["liquid"]["delta_internal_ks_minus_xwm"]["mean"]
            - phase_results["solid"]["delta_internal_ks_minus_xwm"]["mean"]
        )
        corrected_g = xwm_g[temperature] + delta_delta_g
        corrected_h = xwm_h[temperature] + delta_delta_h
        corrected_g_points.append((float(temperature), corrected_g))
        corrected_h_points.append((float(temperature), corrected_h))
        temperature_results[str(temperature)] = {
            "phases": phase_results,
            "oneway_delta_delta_g_ks_minus_xwm_eV_per_atom": delta_delta_g,
            "oneway_delta_delta_g_ks_minus_xwm_meV_per_atom": delta_delta_g * 1000.0,
            "delta_delta_internal_ks_minus_xwm_eV_per_atom": delta_delta_h,
            "delta_delta_internal_ks_minus_xwm_meV_per_atom": delta_delta_h * 1000.0,
            "xwm_delta_g_eV_per_atom": xwm_g[temperature],
            "corrected_delta_g_eV_per_atom": corrected_g,
            "xwm_delta_h_eV_per_atom": xwm_h[temperature],
            "corrected_delta_h_eV_per_atom": corrected_h,
        }

    direct_root = piecewise_root(corrected_g_points)
    integral_root = None
    try:
        integral_root = solve_melting_temperature(
            corrected_g_points[0][0], corrected_g_points[0][1], corrected_h_points
        )
    except ValueError:
        pass

    d50_rows = enthalpy_summary["windows"]["d50"]["points"]
    h_means = {int(row["temperature_k"]): float(row["delta_h_ev_per_atom"]) for row in d50_rows}
    h_errors = {
        int(row["temperature_k"]): float(row["block_standard_error_mev_per_atom"]) / 1000.0
        for row in d50_rows
    }
    anchor_mean = float(xwm_anchor["primary_anchor"]["delta_g_liquid_minus_solid_ev_per_atom"])
    anchor_error = float(xwm_anchor["uncertainty_budget_mev_per_atom"]["statistical_rss"]) / 1000.0
    rng = random.Random(args.seed)
    direct_roots = []
    integral_roots = []
    for _ in range(args.bootstrap_samples):
        sampled_h = {
            temperature: rng.gauss(h_means[temperature], h_errors[temperature])
            for temperature in (850, 950, 1000, 1050)
        }
        sampled_h[900] = 0.5 * (sampled_h[850] + sampled_h[950])
        sampled_anchor = rng.gauss(anchor_mean, anchor_error)
        sampled_xwm_grid = {
            int(temperature): temperature * g_over_t
            for temperature, g_over_t in gibbs_over_temperature(
                900.0, sampled_anchor, sorted((float(t), h) for t, h in sampled_h.items())
            )
        }
        sampled_g_points = []
        sampled_corrected_h = []
        for temperature in (950, 1000, 1050):
            sampled_phase_fep = {}
            sampled_phase_u = {}
            for phase in ("solid", "liquid"):
                source = grouped[(temperature, phase)]
                sample = [rng.choice(source) for _ in source]
                sampled_phase_fep[phase] = fep(
                    [row["delta_mermin_ks_minus_xwm_eV"] for row in sample], temperature
                )[0] / NATOMS
                sampled_phase_u[phase] = statistics.fmean(
                    row["delta_internal_ks_minus_xwm_eV"] / NATOMS for row in sample
                )
            delta_g_correction = sampled_phase_fep["liquid"] - sampled_phase_fep["solid"]
            delta_h_correction = sampled_phase_u["liquid"] - sampled_phase_u["solid"]
            sampled_g_points.append(
                (float(temperature), sampled_xwm_grid[temperature] + delta_g_correction)
            )
            sampled_corrected_h.append(
                (float(temperature), sampled_h[temperature] + delta_h_correction)
            )
        sampled_root = piecewise_root(sampled_g_points)
        if sampled_root is not None:
            direct_roots.append(sampled_root)
        try:
            sampled_root = solve_melting_temperature(
                sampled_g_points[0][0], sampled_g_points[0][1], sampled_corrected_h
            )
        except ValueError:
            sampled_root = None
        if sampled_root is not None:
            integral_roots.append(sampled_root)

    minimum_ess_fraction = min(
        phase["fep_effective_sample_size_fraction"]
        for temperature in temperature_results.values()
        for phase in temperature["phases"].values()
    )
    result = {
        "schema": "mg-xwm-ksdft-multitemperature-oneway-diagnostic-analysis-v1",
        "status": "oneway_diagnostic_complete",
        "reporting_label": "single-direction diagnostic correction from XWM ensembles; no reverse KS ensemble",
        "thermodynamic_convention": "DeltaG and DeltaH are liquid minus solid",
        "computational_validation": {
            "checks": validation_checks,
            "maximum_electronic_iterations": max(row["electronic_iterations"] for row in rows),
            "maximum_highest_band_occupation": max(
                row["maximum_highest_band_occupation"] for row in rows
            ),
            "minimum_nearest_neighbor_A": min(row["nearest_neighbor_A"] for row in rows),
        },
        "temperature_results": temperature_results,
        "overlap_diagnostic": {
            "minimum_fep_effective_sample_size_fraction": minimum_ess_fraction,
            "all_phase_temperature_ess_fraction_ge_0p2": minimum_ess_fraction >= 0.2,
        },
        "roots": {
            "direct_corrected_delta_g_piecewise_root_K": direct_root,
            "corrected_delta_h_integral_root_K": integral_root,
            "direct_delta_g_points": [
                {"temperature_K": t, "delta_g_eV_per_atom": g}
                for t, g in corrected_g_points
            ],
            "corrected_delta_h_points": [
                {"temperature_K": t, "delta_h_eV_per_atom": h}
                for t, h in corrected_h_points
            ],
        },
        "bootstrap": {
            "seed": args.seed,
            "sampling_model": "XWM d50 Gaussian block-SE plus within-ensemble snapshot resampling",
            "direct_delta_g_root": bootstrap_summary(direct_roots, args.bootstrap_samples),
            "corrected_delta_h_integral_root": bootstrap_summary(
                integral_roots, args.bootstrap_samples
            ),
            "systematic_note": "Intervals are conditional statistical diagnostics and exclude the full conservative XWM anchor systematic allowance.",
        },
        "comparison_K": {
            "preliminary_XWM_with_user_authorized_gate_waivers": float(
                xwm_anchor["primary_d50_melting_temperature_k"]
            ),
            "KSDFT_oneway_direct_root": direct_root,
            "KSDFT_oneway_enthalpy_integral_root": integral_root,
            "WT": 986.45,
            "KS_corrected_WT_approximate": 1021.0,
            "experiment": 923.0,
        },
        "gate_waiver": waiver,
        "provenance": {
            "preparation_manifest": {
                "path": str((root / "PREPARATION_MANIFEST.json").resolve()),
                "sha256": sha256(root / "PREPARATION_MANIFEST.json"),
            },
            "preparation_sha256s": {
                "path": str((root / "PREPARATION_SHA256SUMS").resolve()),
                "sha256": sha256(root / "PREPARATION_SHA256SUMS"),
            },
            "xwm_anchor": {"path": str(args.xwm_anchor.resolve()), "sha256": sha256(args.xwm_anchor)},
            "enthalpy_summary": {
                "path": str(args.enthalpy_summary.resolve()),
                "sha256": sha256(args.enthalpy_summary),
            },
            "waiver": {"path": str(args.waiver.resolve()), "sha256": sha256(args.waiver)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "output_sha256": sha256(args.output),
                "minimum_ess_fraction": minimum_ess_fraction,
                "direct_root_K": direct_root,
                "enthalpy_integral_root_K": integral_root,
                "bootstrap_direct": result["bootstrap"]["direct_delta_g_root"],
                "bootstrap_integral": result["bootstrap"]["corrected_delta_h_integral_root"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
