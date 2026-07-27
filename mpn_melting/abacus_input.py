from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .structures import AtomSet, format_vector, repeat_scaled_regions

ANGSTROM_TO_BOHR = 1.8897261254578281


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def input_text(params: Dict[str, object], suffix: str, calculation: str | None = None) -> str:
    merged = dict(params)
    merged["suffix"] = suffix
    if calculation:
        merged["calculation"] = calculation

    lines = ["INPUT_PARAMETERS"]
    order = [
        "suffix",
        "calculation",
        "esolver_type",
        "basis_type",
        "dft_functional",
        "symmetry",
        "pseudo_dir",
        "pseudo_rcut",
        "ecutwfc",
        "ecutrho",
        "scf_nmax",
        "cal_force",
        "cal_stress",
        "of_kinetic",
        "of_method",
        "of_conv",
        "of_tole",
        "of_tolp",
        "of_ml_device",
    ]
    for key in order:
        if key in merged:
            lines.append(f"{key} {format_value(merged[key])}")
    local_only = {
        "abacus_executable",
        "kmesh",
        "mpirun_executable",
        "mpirun_extra_args",
        "mpirun_np",
        "mpn_model",
    }
    for key in sorted(set(merged) - set(order) - local_only):
        lines.append(f"{key} {format_value(merged[key])}")
    return "\n".join(lines) + "\n"


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return " ".join(format_value(item) for item in value)
    return str(value)


def stru_text(atoms: AtomSet, species_configs: Sequence[dict]) -> str:
    by_symbol = {cfg["element"]: cfg for cfg in species_configs}
    lines: List[str] = []
    lines.append("ATOMIC_SPECIES")
    for symbol in atoms.species:
        cfg = by_symbol[symbol]
        fields = [symbol, str(cfg["mass"]), cfg["pseudopotential"]]
        if cfg.get("pseudo_type"):
            fields.append(cfg["pseudo_type"])
        lines.append(" ".join(fields))
    lines.append("")
    lines.append("LATTICE_CONSTANT")
    lines.append(f"{ANGSTROM_TO_BOHR:.16f}")
    lines.append("")
    lines.append("LATTICE_VECTORS")
    for vec in atoms.lattice_vectors:
        lines.append(format_vector(vec))
    lines.append("")
    lines.append("ATOMIC_POSITIONS")
    lines.append("Direct")
    lines.append("")
    for symbol in atoms.species:
        entries = [
            (
                pos,
                atoms.velocities[idx] if atoms.velocities else None,
                atoms.movements[idx] if atoms.movements else (1, 1, 1),
            )
            for idx, (atom_symbol, pos) in enumerate(zip(atoms.symbols, atoms.scaled_positions))
            if atom_symbol == symbol
        ]
        lines.append(symbol)
        lines.append("0.0")
        lines.append(str(len(entries)))
        for pos, velocity, movement in entries:
            line = f"{format_vector(pos)} {movement[0]} {movement[1]} {movement[2]}"
            if velocity is not None:
                line += f" v {format_vector(velocity)}"
            lines.append(line)
    return "\n".join(lines) + "\n"


def kpt_text(kmesh: Iterable[int]) -> str:
    mesh = list(kmesh)
    if len(mesh) != 3:
        raise ValueError("kmesh must have three integers")
    return "\n".join(["K_POINTS", "0", "Gamma", f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0"]) + "\n"


def run_script_text(
    abacus_executable: str = "abacus",
    mpirun_executable: str = "mpirun",
    mpirun_np: int = 1,
    mpirun_extra_args: Iterable[str] = (),
) -> str:
    if mpirun_np and int(mpirun_np) > 1:
        extra = "".join(f' "{argument}"' for argument in mpirun_extra_args)
        return (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'"{mpirun_executable}"{extra} -np {int(mpirun_np)} '
            f'"{abacus_executable}"\n'
        )
    return "#!/usr/bin/env bash\nset -euo pipefail\n" f'"{abacus_executable}"\n'


def write_job(
    out_dir: Path,
    atoms: AtomSet,
    element_config: dict,
    abacus_config: dict,
    job_type: str,
    suffix: str,
    calculation: str | None = None,
    extra_metadata: dict | None = None,
    region_labels: Sequence[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_config = dict(abacus_config)
    if input_config.get("pseudo_dir"):
        input_config["pseudo_dir"] = relative_path_for_input(Path(str(input_config["pseudo_dir"])), out_dir)
    (out_dir / "INPUT").write_text(input_text(input_config, suffix=suffix, calculation=calculation), encoding="utf-8")
    (out_dir / "STRU").write_text(stru_text(atoms, [element_config]), encoding="utf-8")
    (out_dir / "KPT").write_text(kpt_text(abacus_config.get("kmesh", [1, 1, 1])), encoding="utf-8")
    run_script = out_dir / "run_local.sh"
    run_script.write_text(
        run_script_text(
            str(abacus_config.get("abacus_executable", "abacus")),
            str(abacus_config.get("mpirun_executable", "mpirun")),
            int(abacus_config.get("mpirun_np", 1)),
            tuple(abacus_config.get("mpirun_extra_args", ())),
        ),
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)
    if abacus_config.get("mpn_model"):
        shutil.copy2(str(abacus_config["mpn_model"]), out_dir / "net.pt")

    if region_labels is None and job_type == "coexistence_seed":
        region_labels = repeat_scaled_regions(atoms)
    if region_labels:
        write_region_map(out_dir / "regions.csv", atoms.symbols, atoms.scaled_positions, region_labels)

    metadata = {
        "job_type": job_type,
        "element": element_config["element"],
        "structure": element_config["structure"],
        "natoms": atoms.natoms,
        "abacus": {
            "esolver_type": abacus_config.get("esolver_type"),
            "of_kinetic": abacus_config.get("of_kinetic"),
            "of_method": abacus_config.get("of_method"),
            "ecutwfc": abacus_config.get("ecutwfc"),
            "ecutrho": abacus_config.get("ecutrho"),
            "mpn_model": "net.pt" if abacus_config.get("mpn_model") else None,
        },
        "pseudopotential": element_config["pseudopotential"],
        "region_counts": count_regions(region_labels) if region_labels else None,
        "region_map": "regions.csv" if region_labels else None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(out_dir / "metadata.json", metadata)


def relative_path_for_input(path: Path, out_dir: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return os.path.relpath(path, out_dir)
    except ValueError:
        return str(path)


def count_regions(region_labels: Sequence[str] | None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not region_labels:
        return counts
    for label in region_labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def write_region_map(path: Path, symbols: Sequence[str], scaled_positions: Sequence[tuple], labels: Sequence[str]) -> None:
    lines = ["index,symbol,fx,fy,fz,region"]
    for idx, (symbol, pos, label) in enumerate(zip(symbols, scaled_positions, labels), start=1):
        lines.append(f"{idx},{symbol},{pos[0]:.12f},{pos[1]:.12f},{pos[2]:.12f},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
