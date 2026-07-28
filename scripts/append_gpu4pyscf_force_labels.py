#!/usr/bin/env python3
"""Append same-level PySCF/GPU4PySCF force labels to existing zarr labels."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from pyscf import dft
from pyscf.lib.chkfile import load, load_mol
from tqdm import tqdm

from mldft.utils.molecules import build_molecule_np


FORCE_GROUP = "metadata/pbe_derivatives"
FORCE_KEY = f"{FORCE_GROUP}/forces"
_WORKER_GPU_DEVICE_ID: int | None = None
_WORKER_LOCAL_DEVICE_ID: int | None = None
_WORKER_CUDA_VISIBLE_DEVICES: str | None = None


@dataclass(frozen=True)
class ForceConfig:
    backend: str
    xc: str
    basis: str
    initialization: str
    grid_level: int
    prune_method: str
    density_fit_basis: str
    density_fit_threshold: int
    convergence_tolerance: float
    max_cycle: int
    diis_start_cycle: int
    diis_space: int
    pyscf_verbose: int
    geometry_unit: str
    kohn_sham_dir: str | None
    chk_derivatives_mode: str
    overwrite: bool


def _replace_zarr_dataset(group: zarr.Group, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind == "S" and value.shape == ():
        value = value.item().decode()
    if key in group:
        array = group[key]
        value_array = np.asarray(value)
        if array.shape != value_array.shape:
            raise ValueError(
                f"Cannot overwrite {group.path}/{key}: existing shape {array.shape} "
                f"!= new shape {value_array.shape}."
            )
        if array.shape == ():
            array[()] = value
        else:
            array[...] = value
    else:
        group.create_dataset(key, data=value, compressor=None)


def _parse_label_name(path: Path) -> tuple[int, int | None]:
    stem = path.name.removesuffix(".zarr.zip")
    parts = stem.split(".")
    molecule_id = int(parts[0])
    sample_id = int(parts[1]) if len(parts) == 2 else None
    return molecule_id, sample_id


def _expand_label_paths(paths: list[Path]) -> list[Path]:
    labels: list[Path] = []
    for path in paths:
        if path.is_dir():
            labels.extend(path.glob("*.zarr.zip"))
        else:
            labels.append(path)
    return sorted(labels)


def _maybe_to_numpy(value: Any) -> Any:
    module = type(value).__module__.split(".")[0]
    if module == "cupy":
        return value.get()
    if isinstance(value, dict):
        return {key: _maybe_to_numpy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_maybe_to_numpy(item) for item in value)
    if isinstance(value, list):
        return [_maybe_to_numpy(item) for item in value]
    return value


def _read_geometry(label_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    root = zarr.open(label_path, mode="r")
    atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
    atom_pos = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
    charge = int(np.asarray(root["geometry/charge"])) if "geometry/charge" in root else 0
    if "geometry/spin" in root:
        spin = int(np.asarray(root["geometry/spin"]))
    elif "geometry/multiplicity" in root:
        spin = int(np.asarray(root["geometry/multiplicity"])) - 1
    else:
        spin = 0
    return atomic_numbers, atom_pos, charge, spin


def _default_kohn_sham_dir(label_path: Path) -> Path | None:
    label_dir = label_path.parent
    if label_dir.name.startswith("labels"):
        candidate = label_dir.parent / "kohn_sham"
        if candidate.is_dir():
            return candidate
    return None


def _find_chk_file(label_path: Path, kohn_sham_dir: str | None) -> Path | None:
    molecule_id, sample_id = _parse_label_name(label_path)
    search_dir = Path(kohn_sham_dir) if kohn_sham_dir else _default_kohn_sham_dir(label_path)
    if search_dir is None or not search_dir.is_dir():
        return None
    if sample_id is None:
        pattern = f"*_{molecule_id:07d}.chk"
    else:
        pattern = f"*_{molecule_id:07d}.{sample_id:07d}.chk"
    matches = sorted(search_dir.glob(pattern))
    return matches[0] if matches else None


def _compare_geometry_to_chk(
    chk_file: Path,
    atomic_numbers: np.ndarray,
    atom_pos: np.ndarray,
) -> tuple[str, float] | None:
    mol = load_mol(chk_file.as_posix())
    if not np.array_equal(np.asarray(mol.atom_charges(), dtype=np.int64), atomic_numbers):
        return None

    chk_pos_ang = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    chk_pos_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    diff_ang = float(np.max(np.abs(atom_pos - chk_pos_ang)))
    diff_bohr = float(np.max(np.abs(atom_pos - chk_pos_bohr)))
    if diff_bohr < diff_ang:
        return "Bohr", diff_bohr
    return "Angstrom", diff_ang


def _resolve_geometry_unit(
    label_path: Path,
    atomic_numbers: np.ndarray,
    atom_pos: np.ndarray,
    cfg: ForceConfig,
) -> tuple[str, str, float | None]:
    requested_unit = cfg.geometry_unit.lower()
    if requested_unit in ("angstrom", "ang"):
        return "Angstrom", "", None
    if requested_unit in ("bohr", "au", "atomic"):
        return "Bohr", "", None
    if requested_unit != "auto":
        raise ValueError(f"Unknown geometry unit: {cfg.geometry_unit}")

    chk_file = _find_chk_file(label_path, cfg.kohn_sham_dir)
    if chk_file is None:
        return "Angstrom", "", None

    geometry_match = _compare_geometry_to_chk(chk_file, atomic_numbers, atom_pos)
    if geometry_match is None:
        return "Angstrom", chk_file.as_posix(), None

    geometry_unit, geometry_unit_max_diff = geometry_match
    return geometry_unit, chk_file.as_posix(), geometry_unit_max_diff


def _load_chk_derivatives(
    label_path: Path,
    atomic_numbers: np.ndarray,
    atom_pos: np.ndarray,
    cfg: ForceConfig,
    fallback_reason: Exception | None = None,
) -> dict[str, Any] | None:
    chk_file = _find_chk_file(label_path, cfg.kohn_sham_dir)
    if chk_file is None:
        return None
    derivatives = load(chk_file.as_posix(), "Derivatives")
    if derivatives is None or "forces" not in derivatives or "nuclear_gradient" not in derivatives:
        return None

    geometry_match = _compare_geometry_to_chk(chk_file, atomic_numbers, atom_pos)
    if geometry_match is None:
        raise RuntimeError(
            f"Refusing chk derivative fallback for {label_path}: atom identities do not match "
            f"{chk_file}."
        )
    geometry_unit, geometry_unit_max_diff = geometry_match
    if geometry_unit_max_diff > 1e-5:
        raise RuntimeError(
            f"Refusing chk derivative fallback for {label_path}: geometry differs from "
            f"{chk_file} by max abs {geometry_unit_max_diff}."
        )

    forces = np.asarray(derivatives["forces"], dtype=np.float64)
    nuclear_gradient = np.asarray(derivatives["nuclear_gradient"], dtype=np.float64)
    expected_shape = (len(atomic_numbers), 3)
    if forces.shape != expected_shape or nuclear_gradient.shape != expected_shape:
        raise RuntimeError(
            f"Refusing chk derivative fallback for {label_path}: force shape {forces.shape} "
            f"or gradient shape {nuclear_gradient.shape} does not match {expected_shape}."
        )

    results = load(chk_file.as_posix(), "Results")
    return {
        "dft_level": derivatives.get("dft_level", f"{cfg.xc}/{cfg.basis}"),
        "backend": "chk_derivatives",
        "source_backend": cfg.backend,
        "kohn_sham_chk": chk_file.as_posix(),
        "fallback_reason": "" if fallback_reason is None else repr(fallback_reason),
        "geometry_unit": geometry_unit,
        "geometry_unit_inference_chk": chk_file.as_posix(),
        "geometry_unit_inference_max_diff": geometry_unit_max_diff,
        "gpu_device_id": "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "forces_enabled": True,
        "hessian_enabled": bool(derivatives.get("hessian_enabled", False)),
        "nuclear_gradient": nuclear_gradient,
        "forces": forces,
        "total_energy": None if results is None else float(results["total_energy"]),
        "converged": None if results is None else bool(results["converged"]),
    }


def _set_gpu_device(device_id: int | None) -> str | None:
    if device_id is None:
        return None
    import cupy

    cupy.cuda.Device(device_id).use()
    return str(device_id)


def _build_mean_field(mol, cfg: ForceConfig, device_id: int | None):
    mf = dft.RKS(mol, xc=cfg.xc)
    mf.verbose = cfg.pyscf_verbose
    if len(mol.atom_charges()) >= cfg.density_fit_threshold:
        mf = mf.density_fit(cfg.density_fit_basis)
    mf.grids.level = cfg.grid_level

    if cfg.prune_method == "nwchem_prune":
        mf.grids.prune = dft.nwchem_prune
    elif cfg.prune_method in ("none", "None", ""):
        mf.grids.prune = None
    else:
        raise NotImplementedError(f"Unsupported prune method for force appender: {cfg.prune_method}")

    if cfg.backend == "cpu":
        return mf, None
    if cfg.backend != "gpu4pyscf":
        raise ValueError(f"Unknown backend: {cfg.backend}")

    try:
        import gpu4pyscf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GPU4PySCF backend requested but gpu4pyscf is not installed. "
            "Install the CUDA-matched package, for example gpu4pyscf-cuda12x "
            "and cutensor-cu12 on CUDA 12 systems."
        ) from exc

    used_device = _set_gpu_device(device_id)
    return mf.to_gpu(), used_device


def _gradient_kernel(mf) -> np.ndarray:
    try:
        gradient = mf.nuc_grad_method().kernel()
    except AttributeError:
        gradient = mf.Gradients().kernel()
    return np.asarray(_maybe_to_numpy(gradient), dtype=np.float64)


def _compute_force(
    label_path: Path,
    cfg: ForceConfig,
    device_id: int | None,
    assigned_device_id: int | None = None,
) -> dict[str, Any]:
    atomic_numbers, atom_pos, charge, spin = _read_geometry(label_path)
    if cfg.chk_derivatives_mode == "prefer":
        derivatives = _load_chk_derivatives(label_path, atomic_numbers, atom_pos, cfg)
        if derivatives is not None:
            return derivatives

    geometry_unit, chk_file, geometry_unit_max_diff = _resolve_geometry_unit(
        label_path,
        atomic_numbers,
        atom_pos,
        cfg,
    )
    mol = build_molecule_np(
        atomic_numbers,
        atom_pos,
        basis=cfg.basis,
        unit=geometry_unit,
        charge=charge,
        spin=spin,
    )
    mol.verbose = cfg.pyscf_verbose
    mf, used_device = _build_mean_field(mol, cfg, device_id)
    try:
        mf.run(
            init_guess=cfg.initialization,
            max_cycle=cfg.max_cycle,
            conv_tol=cfg.convergence_tolerance,
            diis_start_cycle=cfg.diis_start_cycle,
            diis_space=cfg.diis_space,
        )
        if not mf.converged:
            raise RuntimeError("SCF did not converge.")
        nuclear_gradient = _gradient_kernel(mf)
        if nuclear_gradient.shape != (len(atomic_numbers), 3):
            raise RuntimeError(
                f"Gradient shape {nuclear_gradient.shape} does not match "
                f"expected {(len(atomic_numbers), 3)}."
            )
    except Exception as exc:
        if cfg.chk_derivatives_mode == "fallback":
            derivatives = _load_chk_derivatives(label_path, atomic_numbers, atom_pos, cfg, exc)
            if derivatives is not None:
                if assigned_device_id is not None:
                    derivatives["gpu_device_id"] = str(assigned_device_id)
                return derivatives
        raise

    forces = -nuclear_gradient
    return {
        "dft_level": f"{cfg.xc}/{cfg.basis}",
        "backend": cfg.backend,
        "source_backend": cfg.backend,
        "kohn_sham_chk": "",
        "fallback_reason": "",
        "geometry_unit": geometry_unit,
        "geometry_unit_inference_chk": chk_file,
        "geometry_unit_inference_max_diff": geometry_unit_max_diff,
        "gpu_device_id": (
            ""
            if assigned_device_id is None and used_device is None
            else str(assigned_device_id if assigned_device_id is not None else used_device)
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "forces_enabled": True,
        "hessian_enabled": False,
        "nuclear_gradient": nuclear_gradient,
        "forces": forces,
        "total_energy": float(mf.e_tot),
        "converged": bool(mf.converged),
    }


def _append_force(label_path: Path, derivatives: dict[str, Any]) -> None:
    with zarr.ZipStore(label_path, mode="a") as zipstore:
        root = zarr.open(zipstore, mode="a")
        metadata = root.require_group("metadata")
        group = metadata.require_group("pbe_derivatives")
        for key, value in derivatives.items():
            _replace_zarr_dataset(group, key, value)


def _label_has_force(label_path: Path) -> bool:
    root = zarr.open(label_path, mode="r")
    return FORCE_KEY in root


def _cuda_visible_device_tokens() -> tuple[str, ...] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _cuda_visible_device_for_worker(
    device_id: int,
    cuda_visible_device_tokens: tuple[str, ...] | None,
) -> str:
    if cuda_visible_device_tokens is None:
        return str(device_id)
    if device_id < 0 or device_id >= len(cuda_visible_device_tokens):
        raise ValueError(
            f"GPU device id {device_id} is outside CUDA_VISIBLE_DEVICES "
            f"with {len(cuda_visible_device_tokens)} visible device(s)."
        )
    return cuda_visible_device_tokens[device_id]


def _assign_worker_gpu(
    worker_slot: int,
    gpu_device_ids: tuple[int, ...] | None,
    cuda_visible_device_tokens: tuple[str, ...] | None,
) -> None:
    global _WORKER_CUDA_VISIBLE_DEVICES
    global _WORKER_GPU_DEVICE_ID
    global _WORKER_LOCAL_DEVICE_ID

    if not gpu_device_ids:
        return
    assigned_device_id = gpu_device_ids[worker_slot % len(gpu_device_ids)]
    cuda_visible_devices = _cuda_visible_device_for_worker(
        assigned_device_id,
        cuda_visible_device_tokens,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    _WORKER_GPU_DEVICE_ID = assigned_device_id
    _WORKER_LOCAL_DEVICE_ID = 0
    _WORKER_CUDA_VISIBLE_DEVICES = cuda_visible_devices


def _init_worker_gpu(
    counter: multiprocessing.Value,
    lock: multiprocessing.Lock,
    gpu_device_ids: tuple[int, ...] | None,
    cuda_visible_device_tokens: tuple[str, ...] | None,
) -> None:
    with lock:
        worker_slot = int(counter.value)
        counter.value += 1
    _assign_worker_gpu(worker_slot, gpu_device_ids, cuda_visible_device_tokens)


def _run_one(
    args: tuple[int, Path, ForceConfig],
) -> dict[str, Any]:
    _, label_path, cfg = args
    molecule_id, sample_id = _parse_label_name(label_path)
    record: dict[str, Any] = {
        "path": label_path.as_posix(),
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "status": "unknown",
    }
    try:
        if _label_has_force(label_path) and not cfg.overwrite:
            record["status"] = "skipped_existing"
            return record
        local_device_id = _WORKER_LOCAL_DEVICE_ID if cfg.backend == "gpu4pyscf" else None
        assigned_device_id = _WORKER_GPU_DEVICE_ID if cfg.backend == "gpu4pyscf" else None
        cuda_visible_devices = (
            _WORKER_CUDA_VISIBLE_DEVICES
            if cfg.backend == "gpu4pyscf" and _WORKER_CUDA_VISIBLE_DEVICES is not None
            else os.environ.get("CUDA_VISIBLE_DEVICES", "")
        )
        derivatives = _compute_force(label_path, cfg, local_device_id, assigned_device_id)
        _append_force(label_path, derivatives)
        forces = derivatives["forces"]
        record.update(
            {
                "status": "ok",
                "backend": derivatives["backend"],
                "source_backend": derivatives["source_backend"],
                "gpu_device_id": derivatives["gpu_device_id"],
                "cuda_visible_devices": cuda_visible_devices,
                "geometry_unit": derivatives["geometry_unit"],
                "geometry_unit_inference_chk": derivatives["geometry_unit_inference_chk"],
                "geometry_unit_inference_max_diff": derivatives[
                    "geometry_unit_inference_max_diff"
                ],
                "kohn_sham_chk": derivatives["kohn_sham_chk"],
                "fallback_reason": derivatives["fallback_reason"],
                "natoms": int(forces.shape[0]),
                "force_norm_max": float(np.linalg.norm(forces, axis=1).max()),
                "total_energy": derivatives["total_energy"],
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return record


def _parse_gpu_device_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", type=Path, help="Label files or directories.")
    parser.add_argument("--backend", choices=("gpu4pyscf", "cpu"), default="gpu4pyscf")
    parser.add_argument("--gpu-device-ids", default=os.environ.get("GPU_DEVICE_IDS"))
    parser.add_argument("--num-processes", type=int, default=int(os.environ.get("NUM_PROCESSES", "1")))
    parser.add_argument("--xc", default="PBE")
    parser.add_argument("--basis", default="6-31G(2df,p)")
    parser.add_argument("--initialization", default="minao")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--prune-method", default="nwchem_prune")
    parser.add_argument("--density-fit-basis", default="def2-universal-jfit")
    parser.add_argument("--density-fit-threshold", type=int, default=30)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-cycle", type=int, default=50)
    parser.add_argument("--diis-start-cycle", type=int, default=0)
    parser.add_argument("--diis-space", type=int, default=8)
    parser.add_argument("--pyscf-verbose", type=int, default=0)
    parser.add_argument(
        "--geometry-unit",
        choices=("auto", "angstrom", "bohr"),
        default="auto",
        help=(
            "Unit for geometry/atom_pos. auto uses a matching Kohn-Sham chk file "
            "when available and falls back to angstrom."
        ),
    )
    parser.add_argument(
        "--kohn-sham-dir",
        type=Path,
        help=(
            "Optional directory containing Kohn-Sham .chk files. If omitted, a sibling "
            "kohn_sham directory next to labels is used when present."
        ),
    )
    parser.add_argument(
        "--chk-derivatives-mode",
        choices=("off", "prefer", "fallback"),
        default="off",
        help=(
            "How to use Derivatives/forces from matching Kohn-Sham chk files. "
            "off keeps the run pure PySCF/GPU4PySCF; fallback uses chk derivatives "
            "only after SCF/gradient failure; prefer reads chk derivatives before SCF."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--records-jsonl", type=Path)
    parser.add_argument("--max-labels", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _expand_label_paths(args.labels)
    if args.max_labels is not None:
        labels = labels[: args.max_labels]
    if not labels:
        raise SystemExit("No .zarr.zip labels found.")

    cfg = ForceConfig(
        backend=args.backend,
        xc=args.xc,
        basis=args.basis,
        initialization=args.initialization,
        grid_level=args.grid_level,
        prune_method=args.prune_method,
        density_fit_basis=args.density_fit_basis,
        density_fit_threshold=args.density_fit_threshold,
        convergence_tolerance=args.convergence_tolerance,
        max_cycle=args.max_cycle,
        diis_start_cycle=args.diis_start_cycle,
        diis_space=args.diis_space,
        pyscf_verbose=args.pyscf_verbose,
        geometry_unit=args.geometry_unit,
        kohn_sham_dir=None if args.kohn_sham_dir is None else args.kohn_sham_dir.as_posix(),
        chk_derivatives_mode=args.chk_derivatives_mode,
        overwrite=args.overwrite,
    )
    gpu_device_ids = _parse_gpu_device_ids(args.gpu_device_ids)
    cuda_visible_device_tokens = _cuda_visible_device_tokens()
    tasks = [(idx, label_path, cfg) for idx, label_path in enumerate(labels)]

    if args.backend == "gpu4pyscf" and args.num_processes > 1 and not gpu_device_ids:
        raise SystemExit(
            "Multiple GPU4PySCF processes require --gpu-device-ids, "
            "for example --gpu-device-ids 0,1,2,3."
        )
    if args.backend == "gpu4pyscf" and gpu_device_ids:
        for device_id in gpu_device_ids:
            _cuda_visible_device_for_worker(device_id, cuda_visible_device_tokens)
        if args.num_processes > len(gpu_device_ids):
            raise SystemExit(
                "--num-processes must not exceed the number of --gpu-device-ids for "
                "GPU4PySCF force generation. Use one worker per GPU."
            )

    if args.num_processes == 1:
        if args.backend == "gpu4pyscf":
            _assign_worker_gpu(0, gpu_device_ids, cuda_visible_device_tokens)
        records = [_run_one(task) for task in tqdm(tasks, desc="force labels")]
    else:
        pool_kwargs: dict[str, Any] = {}
        if args.backend == "gpu4pyscf" and gpu_device_ids:
            counter = multiprocessing.Value("i", 0)
            lock = multiprocessing.Lock()
            pool_kwargs = {
                "initializer": _init_worker_gpu,
                "initargs": (counter, lock, gpu_device_ids, cuda_visible_device_tokens),
            }
        with multiprocessing.Pool(args.num_processes, **pool_kwargs) as pool:
            records = list(
                tqdm(
                    pool.imap_unordered(_run_one, tasks),
                    total=len(tasks),
                    desc="force labels",
                )
            )

    if args.records_jsonl is not None:
        args.records_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.records_jsonl.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    failures = [record for record in records if record["status"] == "failed"]
    ok_records = [record for record in records if record["status"] == "ok"]
    skipped = [record for record in records if record["status"] == "skipped_existing"]
    summary = {
        "config": asdict(cfg),
        "checked_files": len(records),
        "written_force_labels": len(ok_records),
        "skipped_existing": len(skipped),
        "failures": failures,
        "force_norm_max": (
            max(record["force_norm_max"] for record in ok_records) if ok_records else None
        ),
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_text)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(summary_text + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
