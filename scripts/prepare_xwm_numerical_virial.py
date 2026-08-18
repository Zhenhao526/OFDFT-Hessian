#!/usr/bin/env python3
"""Prepare paired-volume XWM single points from ABACUS MD_dump frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.structures import AtomSet


STEP_RE = re.compile(r"STEP OF MOLECULAR DYNAMICS:\s*(\d+)")
THERMO_RE = re.compile(
    r"Energy \(Ry\).*?Temperature \(K\)\s*\n\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
    re.S,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temperatures_by_step(path: Path) -> dict[int, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(STEP_RE.finditer(text))
    result: dict[int, float] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        thermo = THERMO_RE.search(text, match.end(), stop)
        if thermo:
            # ABACUS prints observables for MD_dump step n in the block headed n+1.
            result[int(match.group(1)) - 1] = float(thermo.group(4))
    return result


def invert_3x3(matrix: list[tuple[float, float, float]]) -> list[list[float]]:
    a, b, c = matrix
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(det) < 1e-14:
        raise ValueError("singular lattice in MD_dump")
    return [
        [(b[1] * c[2] - b[2] * c[1]) / det, (a[2] * c[1] - a[1] * c[2]) / det, (a[1] * b[2] - a[2] * b[1]) / det],
        [(b[2] * c[0] - b[0] * c[2]) / det, (a[0] * c[2] - a[2] * c[0]) / det, (a[2] * b[0] - a[0] * b[2]) / det],
        [(b[0] * c[1] - b[1] * c[0]) / det, (a[1] * c[0] - a[0] * c[1]) / det, (a[0] * b[1] - a[1] * b[0]) / det],
    ]


def cart_to_frac(cart: tuple[float, float, float], inverse: list[list[float]]) -> tuple[float, float, float]:
    return tuple(sum(cart[j] * inverse[j][i] for j in range(3)) % 1.0 for i in range(3))  # type: ignore[return-value]


def read_md_dump_frames(path: Path, wanted: set[int]) -> dict[int, AtomSet]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: dict[int, AtomSet] = {}
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].startswith("MDSTEP:"):
            cursor += 1
            continue
        step = int(lines[cursor].split(":", 1)[1])
        lattice_constant = float(lines[cursor + 1].split(":", 1)[1].split()[0])
        if lines[cursor + 2].strip() != "LATTICE_VECTORS":
            raise ValueError(f"unexpected MD_dump layout near step {step}")
        lattice = [
            tuple(float(value) * lattice_constant for value in lines[cursor + offset].split()[:3])
            for offset in (3, 4, 5)
        ]
        cursor += 7
        symbols: list[str] = []
        positions: list[tuple[float, float, float]] = []
        while cursor < len(lines) and not lines[cursor].startswith("MDSTEP:"):
            fields = lines[cursor].split()
            if len(fields) >= 5 and fields[0].isdigit():
                symbols.append(fields[1])
                positions.append(tuple(float(value) for value in fields[2:5]))
            cursor += 1
        if step in wanted:
            inverse = invert_3x3(lattice)
            frames[step] = AtomSet(
                symbols=symbols,
                scaled_positions=[cart_to_frac(position, inverse) for position in positions],
                lattice_vectors=lattice,
            )
    missing = wanted - frames.keys()
    if missing:
        raise ValueError(f"missing MD_dump frames in {path}: {sorted(missing)}")
    return frames


def volume(lattice: list[tuple[float, float, float]]) -> float:
    a, b, c = lattice
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def scaled_atoms(atoms: AtomSet, volume_fraction: float) -> AtomSet:
    linear = volume_fraction ** (1.0 / 3.0)
    lattice = [tuple(linear * value for value in row) for row in atoms.lattice_vectors]
    return replace(atoms, lattice_vectors=lattice)


def find_unique(root: Path, name: str) -> Path:
    paths = list(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"expected one {name} under {root}, found {len(paths)}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--element", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--volume-labels", nargs="*", default=[])
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-phase", choices=("solid", "liquid"))
    parser.add_argument("--source-label")
    parser.add_argument("--source-volume-per-atom", type=float)
    parser.add_argument("--frame-start", type=int, default=310)
    parser.add_argument("--frame-stop", type=int, default=595)
    parser.add_argument("--frame-count", type=int, default=20)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    scan = workspace / "runs/mg_three_method_20260804/xwm/zero_pressure_T0900_scan_v2"
    sources = [
        ("solid", "V22p45", 22.45, scan / "node02_import_20260812/solid/V22p45"),
        ("solid", "V22p80", 22.80, scan / "solid/V22p80"),
        ("solid", "V23p15", 23.15, scan / "node06_import_20260812/solid/V23p15"),
        ("liquid", "V23p25", 23.25, scan / "node02_import_20260812/liquid/V23p25"),
        ("liquid", "V23p70", 23.70, scan / "liquid/V23p70"),
        ("liquid", "V24p15", 24.15, scan / "node06_import_20260812/liquid/V24p15"),
    ]
    source_options = (args.source_run, args.source_phase, args.source_label, args.source_volume_per_atom)
    if any(value is not None for value in source_options):
        if not all(value is not None for value in source_options):
            raise ValueError("--source-run, --source-phase, --source-label, and --source-volume-per-atom are required together")
        source_run = args.source_run
        if not source_run.is_absolute():
            source_run = workspace / source_run
        sources = [(args.source_phase, args.source_label, args.source_volume_per_atom, source_run)]
    if args.volume_labels:
        selected = set(args.volume_labels)
        sources = [source for source in sources if source[1] in selected]
        missing = selected - {source[1] for source in sources}
        if missing:
            raise ValueError(f"unknown volume labels: {sorted(missing)}")
    if args.frame_count < 2 or args.frame_stop <= args.frame_start:
        raise ValueError("frame selection requires count >= 2 and stop > start")
    dump_interval = 5
    available = list(range(args.frame_start, args.frame_stop + 1, dump_interval))
    if args.frame_count > len(available):
        raise ValueError("requested more frames than the dump interval provides")
    indexes = [round(index * (len(available) - 1) / (args.frame_count - 1)) for index in range(args.frame_count)]
    steps = [available[index] for index in indexes]
    if len(set(steps)) != args.frame_count:
        raise ValueError(f"frame selection produced duplicate steps: {steps}")

    config = load_json(args.config)
    config.update({"calculation": "scf", "cal_force": 0, "cal_stress": 0})
    element = load_json(args.element)
    lanes = ["node01_a", "node01_b", "node02_a", "node02_b", "node06_a", "node06_b"]
    lane_jobs: dict[str, list[str]] = {lane: [] for lane in lanes}
    manifest: list[dict] = []
    pair_index = 0

    for phase, label, volume_per_atom, source in sources:
        md_dump = find_unique(source, "MD_dump")
        running_log = find_unique(source, "running_md.log")
        temperatures = temperatures_by_step(running_log)
        frames = read_md_dump_frames(md_dump, set(steps))
        for step in steps:
            if step not in temperatures:
                raise ValueError(f"missing temperature for {source} step {step}")
            atoms = frames[step]
            base_volume = volume(atoms.lattice_vectors)
            lane = lanes[pair_index % len(lanes)]
            pair_index += 1
            frame_payload = {
                "source_md_dump_sha256": sha256(md_dump),
                "source_running_log_sha256": sha256(running_log),
                "source_step": step,
                "source_temperature_k": temperatures[step],
                "source_volume_a3": base_volume,
                "source_volume_per_atom_a3": volume_per_atom,
            }
            frame_sha = hashlib.sha256(json.dumps(frame_payload, sort_keys=True).encode()).hexdigest()
            for sign, fraction in (("minus", 1.0 - args.epsilon), ("plus", 1.0 + args.epsilon)):
                rel = Path("jobs") / phase / label / f"step{step:04d}" / sign
                job = out / rel
                write_job(
                    job,
                    scaled_atoms(atoms, fraction),
                    element,
                    config,
                    job_type="xwm_numerical_virial_singlepoint",
                    suffix=f"xwm_virial_{phase}_{label}_{step:04d}_{sign}",
                    calculation="scf",
                    extra_metadata={
                        **frame_payload,
                        "source_run": str(source.relative_to(workspace)),
                        "frame_audit_sha256": frame_sha,
                        "epsilon": args.epsilon,
                        "volume_sign": sign,
                        "scaled_volume_a3": base_volume * fraction,
                        "assigned_lane": lane,
                    },
                )
                rel_text = str(rel)
                lane_jobs[lane].append(rel_text)
                manifest.append({"job": rel_text, "lane": lane, "phase": phase, "volume_label": label, "step": step, "sign": sign})

    (out / "manifests").mkdir()
    for lane, jobs in lane_jobs.items():
        (out / "manifests" / f"{lane}.txt").write_text("\n".join(jobs) + "\n", encoding="utf-8")
    audit = {
        "schema": "xwm-numerical-virial-preparation-v1",
        "epsilon": args.epsilon,
        "selected_steps": steps,
        "frames_per_trajectory": len(steps),
        "source_trajectories": len(sources),
        "singlepoint_jobs": len(manifest),
        "lanes": {lane: len(jobs) for lane, jobs in lane_jobs.items()},
        "jobs": manifest,
    }
    (out / "PREPARATION_MANIFEST.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("frames_per_trajectory", "source_trajectories", "singlepoint_jobs", "lanes")}, indent=2))


if __name__ == "__main__":
    main()
