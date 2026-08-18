#!/usr/bin/env python3
"""Validate the six Mg multitemperature KSDFT pilot single points."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


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


def verify_output_manifest(job: Path) -> dict:
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
    return {"path": str(manifest), "sha256": sha256(manifest), "checked_files": checked}


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


def read_job(job: Path) -> dict:
    if not (job / "sp.done").is_file() or (job / "sp.failed").exists():
        raise ValueError(f"pilot is not complete: {job}")
    if (job / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"nonzero exit code: {job}")
    output_manifest = verify_output_manifest(job)
    logs = sorted(job.glob("OUT.*/running_scf.log"))
    if len(logs) != 1:
        raise ValueError(f"expected one running_scf.log in {job}, found {len(logs)}")
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    energies = [float(value) for value in ENERGY_RE.findall(text)]
    entropy = [(float(ry), float(ev)) for ry, ev in ENTROPY_RE.findall(text)]
    iterations = [int(value) for value in ITER_RE.findall(text)]
    if len(energies) != 1 or len(entropy) != 1 or not iterations:
        raise ValueError(f"incomplete KS output fields in {logs[0]}")
    if FATAL_RE.search(text):
        raise ValueError(f"fatal marker in {logs[0]}")
    occupation_files = sorted(job.glob("OUT.*/eig_occ.txt"))
    if len(occupation_files) != 1:
        raise ValueError(f"expected one eig_occ.txt in {job}")
    highest_band, maximum_occupation = highest_band_occupation(occupation_files[0])
    metadata = json.loads((job / "metadata.json").read_text(encoding="utf-8"))
    mermin = energies[0]
    entropy_ev = entropy[0][1]
    return {
        "job": str(job),
        "metadata_sha256": sha256(job / "metadata.json"),
        "temperature_K": metadata["target_temperature_K"],
        "phase": metadata["phase"],
        "source_step": metadata["source_md_step"],
        "electronic_iterations": iterations[-1],
        "final_etot_is_mermin_eV": mermin,
        "electronic_entropy_minus_ts_eV": entropy_ev,
        "u_ks_eV": mermin - entropy_ev,
        "xwm_potential_energy_eV": metadata["xwm_potential_energy_eV"],
        "delta_mermin_ks_minus_xwm_eV": mermin - metadata["xwm_potential_energy_eV"],
        "delta_internal_ks_minus_xwm_eV": mermin
        - entropy_ev
        - metadata["xwm_potential_energy_eV"],
        "highest_band_index": highest_band,
        "maximum_highest_band_occupation": maximum_occupation,
        "output_manifest": output_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase", choices=("solid", "liquid"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    phases = (args.phase,) if args.phase else ("solid", "liquid")
    rows = []
    for temperature in (950, 1000, 1050):
        for phase in phases:
            rows.append(read_job(root / "jobs" / f"T{temperature:04d}" / phase / "step0775"))
    checks = {
        "expected_job_count": len(rows) == 3 * len(phases),
        "all_iterations_le_150": max(row["electronic_iterations"] for row in rows) <= 150,
        "all_highest_band_indices_eq_192": all(row["highest_band_index"] == 192 for row in rows),
        "all_highest_band_occupations_le_1e-8": max(
            row["maximum_highest_band_occupation"] for row in rows
        )
        <= 1.0e-8,
        "all_output_manifests_verified": True,
    }
    result = {
        "schema": "mg-xwm-ksdft-multitemperature-pilot-validation-v1",
        "status": "pilot_verified" if all(checks.values()) else "pilot_failed",
        "scope": "one-way diagnostic; no reverse KS ensemble",
        "root": str(root),
        "jobs": rows,
        "maximum_electronic_iterations": max(row["electronic_iterations"] for row in rows),
        "maximum_highest_band_occupation": max(
            row["maximum_highest_band_occupation"] for row in rows
        ),
        "checks": checks,
    }
    output = args.output or root / "PILOT_VALIDATION.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256(output),
                "status": result["status"],
                "jobs": len(rows),
                "maximum_electronic_iterations": result["maximum_electronic_iterations"],
                "maximum_highest_band_occupation": result["maximum_highest_band_occupation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
