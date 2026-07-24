#!/usr/bin/env python3
"""Prepare and analyze the Al static benchmarks from Sun and Chen (2024)."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.structures import build_bcc, build_fcc, build_hcp, build_sc

ROOT = Path(__file__).resolve().parents[1]
EV_PER_RY = 13.605693122994
GPA_PER_EV_A3 = 160.21766208
STRUCTURES = ("fcc", "hcp", "bcc", "sc")
VOLUME_FACTORS = (0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10)
HCP_CA_RATIOS = (1.58, 1.61, 1.633, 1.66, 1.69)
REFERENCE_VOLUME_A3 = 4.05**3 / 4.0
FINAL_ENERGY_RE = re.compile(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--mpn-config", default=str(ROOT / "config" / "abacus_mpn_paper_node01.json"))
    prepare.add_argument(
        "--ks-config", default=str(ROOT / "config" / "abacus_ksdft_blps_paper_node01.json")
    )
    prepare.add_argument("--element-config", default=str(ROOT / "config" / "al.json"))
    prepare.set_defaults(func=prepare_suite)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("run_root")
    analyze.set_defaults(func=analyze_suite)
    return result


def atoms_for(structure: str, volume_per_atom: float, ca_ratio: float | None = None):
    if structure == "fcc":
        return build_fcc("Al", (4.0 * volume_per_atom) ** (1.0 / 3.0), (1, 1, 1))
    if structure == "bcc":
        return build_bcc("Al", (2.0 * volume_per_atom) ** (1.0 / 3.0), (1, 1, 1))
    if structure == "sc":
        return build_sc("Al", volume_per_atom ** (1.0 / 3.0), (1, 1, 1))
    if structure == "hcp":
        ratio = float(ca_ratio or math.sqrt(8.0 / 3.0))
        a = (4.0 * volume_per_atom / (math.sqrt(3.0) * ratio)) ** (1.0 / 3.0)
        return build_hcp("Al", a, ratio * a, (1, 1, 1))
    raise ValueError(f"unsupported structure: {structure}")


def prepare_suite(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    element = load_json(Path(args.element_config))
    configs = {"mpn": load_json(Path(args.mpn_config)), "ks": load_json(Path(args.ks_config))}
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing suite: {out}")

    manifest = {
        "paper": "Sun and Chen, Phys. Rev. B 109, 115135 (2024)",
        "doi": "10.1103/PhysRevB.109.115135",
        "ecut_eV": 800.0,
        "ecutwfc_Ry": 800.0 / EV_PER_RY,
        "smearing_eV": 0.1,
        "volume_factors": VOLUME_FACTORS,
        "hcp_ca_ratios": HCP_CA_RATIOS,
        "points": [],
    }
    for solver, base_config in configs.items():
        for structure in STRUCTURES:
            ca_values = HCP_CA_RATIOS if structure == "hcp" else (None,)
            for factor in VOLUME_FACTORS:
                volume = REFERENCE_VOLUME_A3 * factor
                for ca_ratio in ca_values:
                    tag = f"v{factor:.3f}"
                    if ca_ratio is not None:
                        tag += f"_ca{ca_ratio:.3f}"
                    point = out / solver / structure / tag
                    config = dict(base_config)
                    config["kmesh"] = [12, 12, 12] if solver == "ks" and structure == "hcp" else (
                        [20, 20, 20] if solver == "ks" else [1, 1, 1]
                    )
                    atoms = atoms_for(structure, volume, ca_ratio)
                    suffix = f"paper_al_{solver}_{structure}_{tag}".replace(".", "p")
                    write_job(
                        point,
                        atoms,
                        element,
                        config,
                        job_type="paper_static_benchmark",
                        suffix=suffix,
                        calculation="scf",
                        extra_metadata={
                            "solver": solver,
                            "crystal_structure": structure,
                            "volume_factor": factor,
                            "volume_per_atom_A3": volume,
                            "hcp_ca_ratio": ca_ratio,
                            "paper_protocol": True,
                        },
                    )
                    record = {
                        "solver": solver,
                        "structure": structure,
                        "tag": tag,
                        "directory": str(point.relative_to(out)),
                        "natoms": atoms.natoms,
                        "volume_per_atom_A3": volume,
                        "hcp_ca_ratio": ca_ratio,
                    }
                    (point / "point.json").write_text(json.dumps(record, indent=2) + "\n")
                    manifest["points"].append(record)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_runner(out)
    print(f"prepared {len(manifest['points'])} points in {out}")


def write_runner(root: Path) -> None:
    groups = [
        ("mpn", "fcc", "43"),
        ("mpn", "hcp", "44"),
        ("mpn", "bcc", "45"),
        ("mpn", "sc", "46"),
        ("ks", "fcc", "8-15"),
        ("ks", "hcp", "16-23"),
        ("ks", "bcc", "24-31"),
        ("ks", "sc", "32-37"),
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        'root="$(cd "$(dirname "$0")" && pwd)"',
        'status="$root/suite_status.tsv"',
        'printf "timestamp\\tsolver\\tstructure\\tpoint\\tstatus\\n" > "$status"',
        "run_group() {",
        "  local solver=$1 structure=$2 cpus=$3 point rc",
        '  for point in "$root/$solver/$structure"/*; do',
        '    [[ -d "$point" ]] || continue',
        '    if grep -Rqs "!FINAL_ETOT_IS" "$point"/OUT.*/running_scf.log 2>/dev/null; then',
        '      printf "%s\\t%s\\t%s\\t%s\\tcomplete\\n" "$(date -Is)" "$solver" "$structure" "$(basename "$point")" >> "$status"',
        "      continue",
        "    fi",
        '    (cd "$point" && taskset -c "$cpus" ./run_local.sh > run.stdout 2>&1)',
        "    rc=$?",
        '    if [[ $rc -eq 0 ]] && grep -Rqs "!FINAL_ETOT_IS" "$point"/OUT.*/running_scf.log 2>/dev/null; then',
        '      printf "%s\\t%s\\t%s\\t%s\\tcomplete\\n" "$(date -Is)" "$solver" "$structure" "$(basename "$point")" >> "$status"',
        "    else",
        '      printf "%s\\t%s\\t%s\\t%s\\tfailed:%s\\n" "$(date -Is)" "$solver" "$structure" "$(basename "$point")" "$rc" >> "$status"',
        "      return 1",
        "    fi",
        "  done",
        "}",
    ]
    for solver, structure, cpus in groups:
        lines.append(f'run_group "{solver}" "{structure}" "{cpus}" &')
    lines.extend(
        [
            "wait",
            'PYTHONPATH="$root/../../../.." python3 "$root/../../../../scripts/al_mpn_paper_benchmark.py" analyze "$root"',
        ]
    )
    runner = root / "run_suite.sh"
    runner.write_text("\n".join(lines) + "\n")
    runner.chmod(0o755)


def final_energy(point: Path) -> float | None:
    values = []
    for log in point.glob("OUT.*/running_scf.log"):
        values.extend(float(value) for value in FINAL_ENERGY_RE.findall(log.read_text(errors="replace")))
    return values[-1] if values else None


def quadratic_fit(points: list[tuple[float, float]]) -> dict:
    # Fit E(V) = a V^2 + b V + c with the normal equations.
    sums = [sum(v**power for v, _ in points) for power in range(5)]
    rhs = [sum((v**power) * e for v, e in points) for power in range(3)]
    matrix = [
        [sums[4], sums[3], sums[2], rhs[2]],
        [sums[3], sums[2], sums[1], rhs[1]],
        [sums[2], sums[1], sums[0], rhs[0]],
    ]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        for item in range(column, 4):
            matrix[column][item] /= divisor
        for row in range(3):
            if row == column:
                continue
            factor = matrix[row][column]
            for item in range(column, 4):
                matrix[row][item] -= factor * matrix[column][item]
    a, b, c = (matrix[row][3] for row in range(3))
    v0 = -b / (2.0 * a)
    e0 = a * v0 * v0 + b * v0 + c
    bulk_modulus_gpa = 2.0 * a * v0 * GPA_PER_EV_A3
    return {"V0_A3_per_atom": v0, "E0_eV_per_atom": e0, "B0_GPa_quadratic": bulk_modulus_gpa}


def analyze_suite(args: argparse.Namespace) -> None:
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    results = []
    for record in manifest["points"]:
        point = root / record["directory"]
        energy = final_energy(point)
        enriched = dict(record)
        enriched["energy_eV"] = energy
        enriched["energy_eV_per_atom"] = energy / record["natoms"] if energy is not None else None
        results.append(enriched)

    summary = {"completed_points": sum(row["energy_eV"] is not None for row in results), "total_points": len(results)}
    fits = {}
    if summary["completed_points"] == summary["total_points"]:
        for solver in ("mpn", "ks"):
            fits[solver] = {}
            for structure in STRUCTURES:
                rows = [row for row in results if row["solver"] == solver and row["structure"] == structure]
                if structure == "hcp":
                    grouped = {}
                    for row in rows:
                        grouped.setdefault(row["volume_per_atom_A3"], []).append(row)
                    rows = [min(values, key=lambda row: row["energy_eV_per_atom"]) for values in grouped.values()]
                data = sorted((row["volume_per_atom_A3"], row["energy_eV_per_atom"]) for row in rows)
                fits[solver][structure] = quadratic_fit(data)
            fcc_e = fits[solver]["fcc"]["E0_eV_per_atom"]
            for structure in STRUCTURES:
                fits[solver][structure]["delta_E_from_fcc_eV_per_atom"] = (
                    fits[solver][structure]["E0_eV_per_atom"] - fcc_e
                )
    payload = {"summary": summary, "fits": fits, "points": results}
    (root / "benchmark_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
