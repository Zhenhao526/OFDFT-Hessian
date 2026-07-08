from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from .abacus_input import load_json, write_job
from .report import write_analysis_report
from .scan import collect_scan_points, summarize_scan
from .structures import AtomSet, build_coexistence_seed, select_structure
from .trajectory import invert_3x3, matmul_row, parse_md_dump

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
ANGSTROM_PER_FS_PER_ATOMIC_UNIT_VELOCITY = 21.876932361


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mpn-melting")
    sub = parser.add_subparsers(required=True)

    env = sub.add_parser("env-check", help="Check local tools needed by the workflow.")
    env.add_argument("--abacus-config", default=str(CONFIG_DIR / "abacus_mpn.json"))
    env.set_defaults(func=cmd_env_check)

    init = sub.add_parser("init", help="Create run root directories.")
    init.add_argument("--runs", default="runs")
    init.set_defaults(func=cmd_init)

    static = sub.add_parser("make-static", help="Generate a static ABACUS MPN-OFDFT job.")
    add_common_job_args(static)
    static.set_defaults(func=cmd_make_static)

    liquid = sub.add_parser("make-liquid", help="Generate a liquid-preparation MD job template.")
    add_common_job_args(liquid)
    add_md_job_args(liquid, default_steps=10000)
    liquid.add_argument("--source", help="Run directory or MD_dump file used as initial positions.")
    liquid.add_argument("--source-frame", default="last", help="Frame index or 'last' for --source.")
    liquid.set_defaults(func=cmd_make_liquid)

    coexist = sub.add_parser("make-coexist", help="Generate a coexistence seed job template.")
    add_common_job_args(coexist)
    add_md_job_args(coexist, default_steps=20000)
    coexist.add_argument("--liquid-source", help="Run directory or MD_dump file used for the liquid half.")
    coexist.add_argument("--liquid-source-frame", default="last", help="Frame index or 'last' for --liquid-source.")
    coexist.add_argument(
        "--liquid-shift",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("SX", "SY", "SZ"),
        help="Fractional source-cell shift applied to liquid positions before slab mapping.",
    )
    coexist.add_argument("--liquid-displacement", type=float, default=0.35, help="Fallback random liquid-half displacement in Angstrom.")
    coexist.set_defaults(func=cmd_make_coexist)

    restart = sub.add_parser("make-restart", help="Generate an MD restart job from an existing MD_dump frame.")
    add_common_job_args(restart)
    add_md_job_args(restart, default_steps=10000)
    restart.add_argument("--source", required=True, help="Run directory or MD_dump file used as initial positions.")
    restart.add_argument("--source-frame", default="last", help="Frame index or 'last' for --source.")
    restart.add_argument(
        "--region-source",
        help="Run directory or regions.csv whose labels should be preserved. Defaults to --source when possible.",
    )
    restart.set_defaults(func=cmd_make_restart)

    analyze = sub.add_parser("analyze", help="Parse a run directory and write analysis.md.")
    analyze.add_argument("run_dir")
    analyze.set_defaults(func=cmd_analyze)

    scan = sub.add_parser("summarize-scan", help="Summarize a temperature scan and estimate a threshold crossing.")
    scan.add_argument("run_dirs", nargs="+")
    scan.add_argument("--threshold", type=float, default=0.1)
    scan.add_argument("--out")
    scan.set_defaults(func=cmd_summarize_scan)
    return parser


def add_common_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--element", choices=["Al", "Mg"], required=True)
    parser.add_argument("--element-config", help="Override the default element JSON, e.g. for KSDFT UPF pseudopotentials.")
    parser.add_argument("--size", nargs=3, type=int, default=[2, 2, 2], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--abacus-config", default=str(CONFIG_DIR / "abacus_mpn.json"))


def add_md_job_args(parser: argparse.ArgumentParser, default_steps: int) -> None:
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--ensemble", choices=["nvt", "nve"], default="nvt")
    parser.add_argument(
        "--thermostat",
        choices=["nhc", "anderson", "berendsen", "rescaling", "rescale_v", "csvr"],
        default="nhc",
    )
    parser.add_argument("--tfreq", type=float)
    parser.add_argument("--nraise", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--csvr-tau", type=float)
    parser.add_argument("--seed", type=int)


def cmd_env_check(args: argparse.Namespace) -> None:
    tools = ["abacus", "mpirun", "python3", "uv"]
    for tool in tools:
        path = shutil.which(tool)
        status = path if path else "not found"
        print(f"{tool}: {status}")
    config = load_json(Path(args.abacus_config))
    configured_abacus = Path(str(config.get("abacus_executable", "")))
    if configured_abacus:
        status = configured_abacus if configured_abacus.exists() else "not found"
        print(f"abacus(config): {status}")
    pseudo_dir = Path(str(config.get("pseudo_dir", "")))
    if pseudo_dir:
        status = pseudo_dir if pseudo_dir.exists() else "not found"
        print(f"pseudo_dir(config): {status}")
    mpn_model = Path(str(config.get("mpn_model", "")))
    if mpn_model:
        status = mpn_model if mpn_model.exists() else "not found"
        print(f"mpn_model(config): {status}")
    print(f"python: {sys.version.split()[0]}")


def cmd_init(args: argparse.Namespace) -> None:
    runs = Path(args.runs)
    for child in [
        runs / "al" / "static",
        runs / "al" / "liquid",
        runs / "al" / "coexist",
        runs / "mg" / "static",
        runs / "mg" / "liquid",
        runs / "mg" / "coexist",
    ]:
        child.mkdir(parents=True, exist_ok=True)
    print(f"initialized {runs}")


def cmd_make_static(args: argparse.Namespace) -> None:
    element_config = load_element(args.element, args.element_config)
    abacus_config = load_json(Path(args.abacus_config))
    atoms = select_structure(element_config, args.size)
    suffix = f"{args.element.lower()}_static_{args.size[0]}x{args.size[1]}x{args.size[2]}"
    write_job(
        Path(args.out),
        atoms,
        element_config,
        abacus_config,
        job_type="static",
        suffix=suffix,
        calculation="scf",
        extra_metadata={"size": args.size},
    )
    print(f"wrote static job: {args.out} ({atoms.natoms} atoms)")


def cmd_make_liquid(args: argparse.Namespace) -> None:
    element_config = load_element(args.element, args.element_config)
    abacus_config = load_json(Path(args.abacus_config))
    apply_md_options(abacus_config, args)
    source = load_atom_source(args.source, args.source_frame, element_config["element"]) if args.source else None
    atoms = source["atoms"] if source else select_structure(element_config, args.size)
    suffix = f"{args.element.lower()}_liquid_T{int(args.temperature)}"
    metadata = md_metadata(args)
    if source:
        metadata.update(
            {
                "initial_structure_source": source["source"],
                "initial_structure_source_step": source["step"],
            }
        )
    write_job(
        Path(args.out),
        atoms,
        element_config,
        abacus_config,
        job_type="liquid_preparation",
        suffix=suffix,
        calculation="md",
        extra_metadata=metadata,
    )
    print(f"wrote liquid-preparation job: {args.out} ({atoms.natoms} atoms)")


def cmd_make_coexist(args: argparse.Namespace) -> None:
    element_config = load_element(args.element, args.element_config)
    abacus_config = load_json(Path(args.abacus_config))
    apply_md_options(abacus_config, args)
    liquid_source = load_liquid_source(args.liquid_source, args.liquid_source_frame) if args.liquid_source else None
    atoms, region_labels = build_coexistence_seed(
        element_config,
        args.size,
        liquid_scaled_positions=liquid_source["scaled_positions"] if liquid_source else None,
        liquid_shift=args.liquid_shift if liquid_source else None,
        liquid_displacement_angstrom=args.liquid_displacement,
        seed=args.seed,
    )
    suffix = f"{args.element.lower()}_coexist_T{int(args.temperature)}"
    metadata = md_metadata(args)
    metadata.update(
        {
            "method": "solid_liquid_coexistence",
            "note": "Two-phase seed: solid_seed occupies the lower half and liquid_seed occupies the upper half.",
            "liquid_source": liquid_source["source"] if liquid_source else None,
            "liquid_source_step": liquid_source["step"] if liquid_source else None,
            "liquid_shift": args.liquid_shift if liquid_source else None,
            "liquid_displacement_angstrom": args.liquid_displacement if not liquid_source else None,
        }
    )
    write_job(
        Path(args.out),
        atoms,
        element_config,
        abacus_config,
        job_type="coexistence_seed",
        suffix=suffix,
        calculation="md",
        extra_metadata=metadata,
        region_labels=region_labels,
    )
    print(f"wrote coexistence seed job: {args.out} ({atoms.natoms} atoms)")


def cmd_make_restart(args: argparse.Namespace) -> None:
    element_config = load_element(args.element, args.element_config)
    abacus_config = load_json(Path(args.abacus_config))
    apply_md_options(abacus_config, args)
    source = load_atom_source(args.source, args.source_frame, element_config["element"], include_velocities=True)
    if source["atoms"].velocities:
        abacus_config["init_vel"] = 1
    region_path = resolve_region_source(args.region_source or args.source)
    region_labels = load_region_labels(region_path) if region_path else None
    suffix = f"{args.element.lower()}_coexist_T{int(args.temperature)}" if region_labels else f"{args.element.lower()}_restart_T{int(args.temperature)}"
    metadata = md_metadata(args)
    metadata.update(
        {
            "initial_structure_source": source["source"],
            "initial_structure_source_step": source["step"],
            "region_source": region_path.as_posix() if region_path else None,
            "method": "restart_from_md_dump",
        }
    )
    write_job(
        Path(args.out),
        source["atoms"],
        element_config,
        abacus_config,
        job_type="coexistence_restart" if region_labels else "md_restart",
        suffix=suffix,
        calculation="md",
        extra_metadata=metadata,
        region_labels=region_labels,
    )
    print(f"wrote restart job: {args.out} ({source['atoms'].natoms} atoms)")


def cmd_analyze(args: argparse.Namespace) -> None:
    report = write_analysis_report(Path(args.run_dir))
    print(f"wrote {report}")


def cmd_summarize_scan(args: argparse.Namespace) -> None:
    points = collect_scan_points(Path(run_dir) for run_dir in args.run_dirs)
    text = summarize_scan(points, threshold=args.threshold)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


def load_element(element: str, element_config: str | None = None) -> dict:
    if element_config:
        return load_json(Path(element_config))
    filename = "al.json" if element == "Al" else "mg.json"
    return load_json(CONFIG_DIR / filename)


def load_liquid_source(source: str, frame_selector: str) -> dict:
    path = Path(source)
    dump = path if path.is_file() else first_md_dump(path)
    if dump is None:
        raise FileNotFoundError(f"no MD_dump found in liquid source: {source}")
    frames = parse_md_dump(dump)
    if not frames:
        raise ValueError(f"no frames parsed from {dump}")
    if frame_selector == "last":
        frame = frames[-1]
    else:
        frame = frames[int(frame_selector)]
    reference = frames[0]
    inv_lattice = invert_3x3(reference.lattice)
    positions = unwrap_against_reference(reference.positions, frame.positions, reference.lattice, inv_lattice)
    scaled_positions = [matmul_row(position, inv_lattice) for position in positions]
    return {
        "source": dump.as_posix(),
        "step": frame.step,
        "scaled_positions": scaled_positions,
    }


def load_atom_source(source: str, frame_selector: str, element: str, include_velocities: bool = False) -> dict:
    path = Path(source)
    dump = path if path.is_file() else first_md_dump(path)
    if dump is None:
        raise FileNotFoundError(f"no MD_dump found in source: {source}")
    frames = parse_md_dump(dump)
    if not frames:
        raise ValueError(f"no frames parsed from {dump}")
    frame = frames[-1] if frame_selector == "last" else frames[int(frame_selector)]
    inv_lattice = invert_3x3(frame.lattice)
    scaled_positions = []
    for position in frame.positions:
        scaled = matmul_row(position, inv_lattice)
        scaled_positions.append(tuple(component % 1.0 for component in scaled))
    velocities = None
    if include_velocities and frame.velocities:
        velocities = [
            tuple(component / ANGSTROM_PER_FS_PER_ATOMIC_UNIT_VELOCITY for component in velocity)
            for velocity in frame.velocities
        ]
    return {
        "source": dump.as_posix(),
        "step": frame.step,
        "atoms": AtomSet(
            symbols=[element] * len(scaled_positions),
            scaled_positions=scaled_positions,
            lattice_vectors=list(frame.lattice),
            velocities=velocities,
        ),
    }


def first_md_dump(run_dir: Path) -> Path | None:
    for path in sorted(run_dir.rglob("MD_dump"), key=lambda item: item.relative_to(run_dir).as_posix()):
        return path
    return None


def resolve_region_source(source: str) -> Path | None:
    path = Path(source)
    if path.is_file():
        if path.name == "regions.csv":
            return path
        candidate = path.parent.parent / "regions.csv"
        return candidate if candidate.exists() else None
    candidate = path / "regions.csv"
    return candidate if candidate.exists() else None


def load_region_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["region"] for row in reader]


def unwrap_against_reference(reference_positions, positions, lattice, inv_lattice):
    unwrapped = []
    for reference, position in zip(reference_positions, positions):
        delta = tuple(position[axis] - reference[axis] for axis in range(3))
        delta_frac = matmul_row(delta, inv_lattice)
        wrapped_frac = tuple(component - round(component) for component in delta_frac)
        wrapped_cart = matmul_row(wrapped_frac, lattice)
        unwrapped.append(tuple(reference[axis] + wrapped_cart[axis] for axis in range(3)))
    return unwrapped


def apply_md_options(abacus_config: dict, args: argparse.Namespace) -> None:
    abacus_config.update(
        {
            "calculation": "md",
            "md_type": args.ensemble,
            "md_nstep": args.steps,
            "md_dt": args.dt,
            "md_tfirst": args.temperature,
            "md_tlast": args.temperature,
        }
    )
    if args.ensemble == "nvt":
        abacus_config["md_thermostat"] = args.thermostat
    optional_keys = {
        "tfreq": "md_tfreq",
        "nraise": "md_nraise",
        "tolerance": "md_tolerance",
        "csvr_tau": "md_csvr_tau",
        "seed": "md_seed",
    }
    for arg_name, input_name in optional_keys.items():
        value = getattr(args, arg_name)
        if value is not None:
            abacus_config[input_name] = value


def md_metadata(args: argparse.Namespace) -> dict:
    metadata = {
        "size": args.size,
        "target_temperature_k": args.temperature,
        "ensemble": args.ensemble,
        "steps": args.steps,
        "dt_fs": args.dt,
        "thermostat": args.thermostat if args.ensemble == "nvt" else None,
    }
    for key in ("tfreq", "nraise", "tolerance", "csvr_tau", "seed"):
        value = getattr(args, key)
        if value is not None:
            metadata[key] = value
    return metadata


if __name__ == "__main__":
    main()
