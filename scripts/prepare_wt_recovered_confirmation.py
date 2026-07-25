#!/usr/bin/env python3
"""Wrap a recovered WT phase pair as an auditable confirmation source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mpn_melting.cli import load_atom_source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_md_dump(run: Path) -> Path:
    candidates = sorted(run.glob("OUT.*/MD_dump"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one MD_dump below {run}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def prepare_recovered_confirmation(
    source_root: Path,
    out: Path,
    *,
    temperature_k: float,
    steps: int,
    expected_last_step: int,
    solid_volume_per_atom_a3: float,
    liquid_volume_per_atom_a3: float,
    expected_sha256: dict[str, str],
    expected_natoms: int = 108,
    csvr_tau_fs: float = 5.0,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if set(expected_sha256) != {"solid", "liquid"}:
        raise ValueError("expected SHA256 values are required for solid and liquid")
    if steps <= 0 or expected_last_step < 0:
        raise ValueError("steps and expected last step must be non-negative")

    volumes = {
        "solid": float(solid_volume_per_atom_a3),
        "liquid": float(liquid_volume_per_atom_a3),
    }
    phases: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for phase in ("solid", "liquid"):
        run = (source_root / phase).resolve()
        if not run.is_dir():
            raise FileNotFoundError(f"missing recovered {phase} run: {run}")
        md_dump = find_md_dump(run)
        actual_sha256 = sha256_file(md_dump)
        if actual_sha256 != expected_sha256[phase]:
            raise ValueError(
                f"{phase} MD_dump SHA256 mismatch: "
                f"{actual_sha256} != {expected_sha256[phase]}"
            )
        source = load_atom_source(
            str(run), "last", "Al", include_velocities=True
        )
        atoms = source["atoms"]
        if atoms.natoms != expected_natoms:
            raise ValueError(
                f"{phase} recovered atom count is {atoms.natoms}, "
                f"expected {expected_natoms}"
            )
        if int(source["step"]) != expected_last_step:
            raise ValueError(
                f"{phase} recovered last step is {source['step']}, "
                f"expected {expected_last_step}"
            )
        if atoms.velocities is None or len(atoms.velocities) != atoms.natoms:
            raise ValueError(f"{phase} recovered last frame has no complete velocities")
        phases.append(
            {
                "phase": phase,
                "run": str(run),
                "source": str(md_dump),
                "source_step": int(source["step"]),
                "volume_per_atom_A3": volumes[phase],
                "source_md_dump_sha256": actual_sha256,
            }
        )
        checks[phase] = {
            "run_exists": True,
            "md_dump_sha256_matches": True,
            "last_step_matches": True,
            "velocities_present": True,
            "natoms": atoms.natoms,
        }

    manifest = {
        "schema": "wt-zero-pressure-confirmation-v1",
        "recovery_schema": "wt-recovered-confirmation-wrapper-v1",
        "target_kedf": "wt",
        "temperature_K": float(temperature_k),
        "target_pressure_kbar": 0.0,
        "steps": int(steps),
        "csvr_tau_fs": float(csvr_tau_fs),
        "source_root": str(source_root),
        "source_velocities_discarded": False,
        "phases": phases,
    }
    preflight = {
        "schema": "wt-recovered-confirmation-preflight-v1",
        "status": "verified",
        "expected_last_step": int(expected_last_step),
        "expected_natoms": int(expected_natoms),
        "checks": checks,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent))
    )
    try:
        for phase in ("solid", "liquid"):
            os.symlink(source_root / phase, temporary / phase, target_is_directory=True)
        (temporary / "confirmation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "recovery_preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(out)
    except Exception:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1100.0)
    parser.add_argument("--steps", type=int, default=1700)
    parser.add_argument("--expected-last-step", type=int, default=1695)
    parser.add_argument("--expected-natoms", type=int, default=108)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--solid-volume", type=float, required=True)
    parser.add_argument("--liquid-volume", type=float, required=True)
    parser.add_argument("--solid-sha256", required=True)
    parser.add_argument("--liquid-sha256", required=True)
    args = parser.parse_args()
    result = prepare_recovered_confirmation(
        args.source_root,
        args.out,
        temperature_k=args.temperature,
        steps=args.steps,
        expected_last_step=args.expected_last_step,
        expected_natoms=args.expected_natoms,
        solid_volume_per_atom_a3=args.solid_volume,
        liquid_volume_per_atom_a3=args.liquid_volume,
        expected_sha256={
            "solid": args.solid_sha256,
            "liquid": args.liquid_sha256,
        },
        csvr_tau_fs=args.csvr_tau,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
