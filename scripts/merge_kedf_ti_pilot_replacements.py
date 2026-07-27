#!/usr/bin/env python3
"""Merge selected TI pilot-window replacements and rerun strict diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from mpn_melting.free_energy import exponential_free_energy_difference
from scripts.analyze_two_phase_run import parse_md_log, series_stats
from scripts.prepare_al108_ti_windows import parse_components, trapezoid


def parse_replacement_specs(specs: list[str]) -> dict[tuple[str, str], Path]:
    replacements: dict[tuple[str, str], Path] = {}
    for spec in specs:
        key, separator, path = spec.partition("=")
        phase, phase_separator, label = key.partition(":")
        if (
            not separator
            or not phase_separator
            or phase not in {"solid", "liquid"}
            or not label
            or not path
        ):
            raise ValueError(
                "--replacement must use solid:lambda_LABEL=PATH or "
                "liquid:lambda_LABEL=PATH"
            )
        replacement_key = (phase, label)
        if replacement_key in replacements:
            raise ValueError(f"duplicate replacement for {phase}:{label}")
        replacements[replacement_key] = Path(path).resolve()
    return replacements


def select_window_runs(
    base_root: Path,
    replacements: dict[tuple[str, str], Path],
) -> dict[str, list[dict]]:
    selected: dict[str, list[dict]] = {}
    known_keys: set[tuple[str, str]] = set()
    for phase in ("solid", "liquid"):
        manifest = json.loads((base_root / phase / "manifest.json").read_text())
        rows = []
        for window in manifest["windows"]:
            key = (phase, str(window["label"]))
            known_keys.add(key)
            run = replacements.get(key, base_root / phase / window["label"])
            rows.append({**window, "run": run.resolve(), "replaced": key in replacements})
        selected[phase] = rows
    unknown = sorted(set(replacements) - known_keys)
    if unknown:
        rendered = ", ".join(f"{phase}:{label}" for phase, label in unknown)
        raise ValueError(f"replacement labels not present in base pilot: {rendered}")
    return selected


def analyze_phase_grid(
    base_root: Path,
    phase: str,
    selected: list[dict],
    temperature_tolerance_K: float,
) -> dict:
    manifest = json.loads((base_root / phase / "manifest.json").read_text())
    expected_steps = int(manifest["steps"])
    target_kedf = str(manifest["target_kedf"]).lower()
    expected_phase_status = f"{phase}_verified"
    rows = []
    raw_by_lambda: dict[float, list[float]] = {}

    for window in selected:
        run = Path(window["run"])
        metadata = json.loads((run / "metadata.json").read_text())
        phase_analysis = json.loads((run / "phase_analysis.json").read_text())
        logs = sorted(run.glob("OUT.*/running_md.log"))
        if not logs:
            raise FileNotFoundError(f"missing running_md.log below {run}")
        components = parse_components(logs[-1])
        md_rows, max_step = parse_md_log(logs[-1])
        late_components = components[len(components) // 2 :]
        late_md = md_rows[len(md_rows) // 2 :]
        delta_values = [row["delta_U_eV"] for row in late_components]
        temperature_stats = series_stats(
            [row["temperature_K"] for row in late_md]
        )
        nearest_neighbor = min(
            (row["nearest_neighbor_A"] for row in components), default=None
        )
        checks = {
            "reached_requested_step": max_step >= expected_steps,
            "component_samples_complete": len(components) >= expected_steps,
            "target_kedf_matches": (
                str(metadata.get("target_kedf", "")).lower() == target_kedf
            ),
            "phase_matches": phase_analysis.get("status") == expected_phase_status,
            "temperature_within_tolerance": (
                abs(temperature_stats["mean"] - manifest["target_temperature_K"])
                <= temperature_tolerance_K
            ),
            "nearest_neighbor_gt_2_A": (
                nearest_neighbor is not None and nearest_neighbor > 2.0
            ),
        }
        raw_by_lambda[window["lambda"]] = delta_values
        rows.append(
            {
                "label": window["label"],
                "lambda": window["lambda"],
                "run": str(run),
                "replaced": window["replaced"],
                "max_step": max_step,
                "component_samples": len(components),
                "phase_status": phase_analysis.get("status"),
                "temperature_last_half_K": temperature_stats,
                "minimum_pair_distance_A": nearest_neighbor,
                "delta_U_last_half_eV": series_stats(delta_values),
                "delta_U_last_half_meV_per_atom": {
                    "mean": statistics.mean(delta_values)
                    * 1000.0
                    / manifest["natoms"],
                    "sd": statistics.stdev(delta_values)
                    * 1000.0
                    / manifest["natoms"]
                    if len(delta_values) > 1
                    else 0.0,
                },
                "checks": checks,
                "status": "verified" if all(checks.values()) else "failed",
            }
        )

    ordered = sorted(rows, key=lambda row: row["lambda"])
    overlaps = []
    temperature = float(manifest["target_temperature_K"])
    for left, right in zip(ordered, ordered[1:]):
        delta_lambda = right["lambda"] - left["lambda"]
        forward = exponential_free_energy_difference(
            [delta_lambda * value for value in raw_by_lambda[left["lambda"]]],
            temperature,
        )
        reverse = exponential_free_energy_difference(
            [-delta_lambda * value for value in raw_by_lambda[right["lambda"]]],
            temperature,
        )
        overlaps.append(
            {
                "lambda_left": left["lambda"],
                "lambda_right": right["lambda"],
                "forward_effective_sample_fraction": forward[
                    "effective_sample_fraction"
                ],
                "reverse_effective_sample_fraction": reverse[
                    "effective_sample_fraction"
                ],
            }
        )

    delta_f = trapezoid(ordered)
    all_windows_verified = all(row["status"] == "verified" for row in rows)
    result = {
        "phase": phase,
        "target_kedf": target_kedf,
        "target_temperature_K": temperature,
        "temperature_tolerance_K": temperature_tolerance_K,
        "natoms": manifest["natoms"],
        "base_manifest": str((base_root / phase / "manifest.json").resolve()),
        "window_results": rows,
        "adjacent_overlap": overlaps,
        "minimum_adjacent_effective_sample_fraction": min(
            min(
                item["forward_effective_sample_fraction"],
                item["reverse_effective_sample_fraction"],
            )
            for item in overlaps
        ),
        "delta_F_target_minus_reference_eV": delta_f,
        "delta_F_target_minus_reference_meV_per_atom": (
            delta_f * 1000.0 / manifest["natoms"]
        ),
        "status": "verified" if all_windows_verified else "failed",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replacement", action="append", default=[])
    parser.add_argument("--temperature-tolerance", type=float, default=20.0)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    base = args.base.resolve()
    replacements = parse_replacement_specs(args.replacement)
    selected = select_window_runs(base, replacements)
    phase_results = {
        phase: analyze_phase_grid(
            base, phase, selected[phase], args.temperature_tolerance
        )
        for phase in ("solid", "liquid")
    }
    status = (
        "pilot_grid_verified"
        if all(result["status"] == "verified" for result in phase_results.values())
        else "pilot_grid_failed"
    )
    result = {
        "schema": "kedf-pair-ti-pilot-replacements-v1",
        "base": str(base),
        "replacement_provenance": {
            f"{phase}:{label}": str(path)
            for (phase, label), path in sorted(replacements.items())
        },
        "phase_results": phase_results,
        "delta_delta_F_liquid_minus_solid_meV_per_atom": (
            phase_results["liquid"][
                "delta_F_target_minus_reference_meV_per_atom"
            ]
            - phase_results["solid"][
                "delta_F_target_minus_reference_meV_per_atom"
            ]
        ),
        "status": status,
    }
    out.mkdir(parents=True)
    (out / "merged_pilot_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
