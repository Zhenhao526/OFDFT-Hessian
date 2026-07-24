#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from mpn_melting.pair_reference import (
    PairVirialResult,
    evaluate_pair_virial,
    pair_pressure_kbar,
)
from mpn_melting.trajectory import parse_md_dump
from scripts.analyze_two_phase_run import complete_trajectory_frames, parse_md_log


PAIR_COMPONENT_RE = re.compile(
    r"PAIR_REFERENCE_COMPONENTS\s+step=(?P<step>\d+)"
    r"\s+U_REF_eV=(?P<energy>[-+0-9.eE]+)"
    r"\s+W_REF_eV=(?P<virial>[-+0-9.eE]+)"
    r"\s+RMIN_A=(?P<nearest>[-+0-9.eE]+)"
)


def parse_logged_virials(path: Path) -> dict[int, PairVirialResult]:
    samples: dict[int, PairVirialResult] = {}
    for match in PAIR_COMPONENT_RE.finditer(path.read_text(encoding="utf-8")):
        samples[int(match.group("step"))] = PairVirialResult(
            potential_energy_ev=float(match.group("energy")),
            configurational_virial_ev=float(match.group("virial")),
            nearest_neighbor_angstrom=float(match.group("nearest")),
        )
    return samples


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tail-frames", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    model = None
    if args.model is not None:
        model_document = json.loads(args.model.read_text(encoding="utf-8"))
        model = model_document["model"]
    log = next(args.run_dir.glob("OUT.*/running_md.log"))
    dump = next(args.run_dir.glob("OUT.*/MD_dump"))
    rows, max_step = parse_md_log(log)
    temperature_by_step = {row["step"]: row["temperature_K"] for row in rows}
    parsed = parse_md_dump(dump)
    frames, dropped = complete_trajectory_frames(parsed)
    frames = frames[-args.tail_frames :]
    if not frames:
        raise ValueError("no complete trajectory frames")

    logged_virials = parse_logged_virials(log)
    metadata_path = args.run_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    target_temperature = metadata.get("target_temperature_K")
    samples = []
    for frame in frames:
        temperature = temperature_by_step.get(frame.step)
        if temperature is None:
            if rows:
                closest = min(rows, key=lambda row: abs(row["step"] - frame.step))
                temperature = closest["temperature_K"]
            elif target_temperature is not None:
                temperature = float(target_temperature)
            else:
                raise ValueError(f"no temperature available for frame {frame.step}")
        virial = logged_virials.get(frame.step)
        virial_source = "log"
        if virial is None:
            if model is None:
                raise ValueError(
                    f"frame {frame.step} has no logged virial; provide --model for fallback"
                )
            virial = evaluate_pair_virial(frame.positions, frame.lattice, model)
            virial_source = "trajectory_recalculation"
        samples.append(
            {
                "step": frame.step,
                "temperature_K": temperature,
                "pressure_kbar": pair_pressure_kbar(
                    virial, len(frame.positions), temperature, frame.lattice
                ),
                "potential_energy_eV": virial.potential_energy_ev,
                "configurational_virial_eV": virial.configurational_virial_ev,
                "nearest_neighbor_A": virial.nearest_neighbor_angstrom,
                "virial_source": virial_source,
            }
        )

    result = {
        "schema": "pair-reference-pressure-analysis-v1",
        "run_dir": str(args.run_dir.resolve()),
        "model": str(args.model.resolve()) if args.model is not None else None,
        "max_md_step": max_step,
        "parsed_frames": len(parsed),
        "dropped_incomplete_frames": dropped,
        "analyzed_frames": len(samples),
        "temperature_K": stats([sample["temperature_K"] for sample in samples]),
        "pressure_kbar": stats([sample["pressure_kbar"] for sample in samples]),
        "nearest_neighbor_A": stats(
            [sample["nearest_neighbor_A"] for sample in samples]
        ),
        "samples": samples,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
