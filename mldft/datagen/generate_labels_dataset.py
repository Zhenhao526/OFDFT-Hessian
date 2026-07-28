"""Compute labels for molecules in the dataset and save them as zarr.zip files.

Can be run with the `mldft_labelgen` command.
"""

import multiprocessing
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import zarr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pyscf.lib.chkfile import load as load_chk_group
from tqdm import tqdm

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.datagen.kohn_sham_dataset import check_config
from mldft.datagen.methods.label_generation import str_to_calculation_fct
from mldft.datagen.methods.save_labels_in_zarr_file import save_density_fitted_data
from mldft.utils import extras
from mldft.utils.molecules import build_molecule_np, construct_aux_mol, load_scf
from mldft.utils.multiprocess import (
    configure_max_memory_per_process,
    configure_processes_and_threads,
    unpack_args_for_imap,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula


def get_id_and_sample_id_from_chk_file(file: Path) -> tuple[int, int | None]:
    """Get the molecule id and sample id from the .chk file.

    Args:
        file: The .chk file.

    Returns:
        Tuple of molecule id and sample id.
    """
    split_filename = file.stem.split("_")[-1].split(".")
    if len(split_filename) == 2:
        molecule_id, sample_id = int(split_filename[0]), int(split_filename[1])
    else:
        molecule_id = int(split_filename[0])
        sample_id = None
    return molecule_id, sample_id


def get_zarr_file_path(label_dir: Path, molecule_id: int, sample_id: int | None = None) -> Path:
    if sample_id is not None:
        return label_dir / f"{molecule_id:07}.{sample_id:07}.zarr.zip"
    else:
        return label_dir / f"{molecule_id:07}.zarr.zip"


def _replace_zarr_dataset(group: zarr.Group, key: str, value):
    if value is None:
        return
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind == "S" and value.shape == ():
        value = value.item().decode()
    if key in group:
        del group[key]
    group.create_dataset(key, data=value, compressor=None)


def append_kohn_sham_derivatives(label_path: Path, chk_file: Path) -> None:
    """Append optional same-level PySCF force/Hessian results from the chk file to labels."""
    try:
        derivatives = load_chk_group(chk_file, "Derivatives")
    except (KeyError, OSError):
        derivatives = None
    try:
        derivative_errors = load_chk_group(chk_file, "DerivativeErrors")
    except (KeyError, OSError):
        derivative_errors = None
    if derivatives is None and derivative_errors is None:
        return

    with zarr.ZipStore(label_path, mode="a") as zipstore:
        root = zarr.open(zipstore, mode="a")
        metadata = root.require_group("metadata")
        if derivatives is not None:
            group = metadata.require_group("pbe_derivatives")
            for key, value in derivatives.items():
                _replace_zarr_dataset(group, key, value)
        if derivative_errors is not None:
            group = metadata.require_group("pbe_derivative_errors")
            for key, value in derivative_errors.items():
                _replace_zarr_dataset(group, key, value)


@unpack_args_for_imap
def run_labelgen_task(
    dataset: DataGenDataset,
    chk_file: Path,
    calculation_fct: callable,
    orbital_basis: str,
    kohn_sham_basis: str,
    of_basis_set: str,
):
    """Run the label generation task for a single molecule."""
    molecule_id, sample_id = get_id_and_sample_id_from_chk_file(chk_file)
    try:
        # Load charges and build molecule in a way similar to the Kohn-Sham logic.
        charges, positions, charge, spin = dataset.load_sample(molecule_id, sample_id)
        mol_orbital_basis = build_molecule_np(
            charges,
            positions,
            basis=orbital_basis,
            unit="Angstrom",
            charge=charge,
            spin=spin,
        )
        mol_density_basis = construct_aux_mol(
            mol_orbital_basis, aux_basis_name=of_basis_set, unit="Bohr", spin=spin
        )
        logger.info(
            f"Computing molecule {molecule_id} {mole_to_sum_formula(mol_orbital_basis, True)} "
            f"with {len(mol_orbital_basis.atom_charges())} atoms."
        )

        label_path = dataset.get_label_file_from_id(molecule_id, sample_id)
        results, initialization, data_of_iteration = load_scf(chk_file)
        data = calculation_fct(
            results=results,
            initialization=initialization,
            data_of_iteration=data_of_iteration,
            mol_orbital_basis=mol_orbital_basis,
            mol_density_basis=mol_density_basis,
        )
        save_density_fitted_data(
            label_path,
            mol_density_basis,
            kohn_sham_basis=kohn_sham_basis,
            path_to_basis_info=(chk_file.parent / "basis_info.npz").as_posix(),
            molecular_data=data,
            mol_id=chk_file.stem,
        )
        append_kohn_sham_derivatives(label_path, chk_file)
        dataset.save_reference_metadata(label_path, molecule_id, sample_id)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.exception(f"Label generation failed for {chk_file}: {e}")


def save_dataset_info(cfg: DictConfig, label_dir: Path):
    """Save relevant config information for the dataset in a yaml file."""
    keys_to_save = ("kohn_sham", "density_fitting_method", "of_basis_set", "dataset")
    dataset_info = {key: cfg.get(key) for key in keys_to_save}
    OmegaConf.save(config=dataset_info, f=label_dir / "dataset_info.yaml")


def run_label_generation(cfg: DictConfig):
    """Run the label generation for the dataset."""
    # Setup logger to log via tqdm and to the configured log file
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        level=cfg.get("log_level", "INFO"),
        colorize=True,
        enqueue=True,
    )

    num_processes = cfg.get("num_processes")
    num_threads_per_process = cfg.get("num_threads_per_process")

    # Instantiate Dataset
    dataset_settings = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_settings["num_processes"] = num_processes
    dataset = hydra.utils.instantiate(dataset_settings)
    kohn_sham_settings = cfg.kohn_sham
    check_config(cfg.kohn_sham, dataset.kohn_sham_data_dir / "config.yaml")
    orbital_basis = kohn_sham_settings.basis

    # Save dataset info (config) in the directory above the labels directory
    save_dataset_info(cfg, dataset.label_dir.parent)

    density_fitting_method = cfg.get("density_fitting_method")
    of_basis_set = cfg.get("of_basis_set")
    ids_this_run = dataset.get_ids_todo_labelgen(
        start_idx=cfg.start_idx, max_num_molecules=cfg.n_molecules
    )
    chk_files = dataset.get_all_chk_files_from_ids(ids_this_run)

    calculation_fct_name = cfg.label_src.calculation_fct
    calculation_fct = str_to_calculation_fct[calculation_fct_name]
    calculation_fct = partial(calculation_fct, density_fitting_method=density_fitting_method)

    num_processes = min(num_processes, len(chk_files))
    if num_processes <= 1:
        if num_threads_per_process == 1:
            logger.warning("Running in single process mode using 1 thread, this may take a while")
        else:
            logger.info(f"Running in single process mode using {num_threads_per_process} threads.")
        for chk_file in tqdm(chk_files, desc="Label Generation", dynamic_ncols=True):
            run_labelgen_task(
                dataset,
                chk_file,
                calculation_fct,
                orbital_basis,
                kohn_sham_settings.basis,
                of_basis_set,
            )
    else:
        logger.info(
            f"Using {num_processes} processes with {num_threads_per_process} threads each."
        )
        args_list = [
            (
                dataset,
                chk_file,
                calculation_fct,
                orbital_basis,
                kohn_sham_settings.basis,
                of_basis_set,
            )
            for chk_file in chk_files
        ]
        with multiprocessing.Pool(processes=num_processes) as pool:
            imap = pool.imap_unordered(run_labelgen_task, args_list)
            for _ in tqdm(
                imap,
                total=len(chk_files),
                desc="Label Generation",
                dynamic_ncols=True,
                smoothing=0,
            ):
                pass

    logger.info("Label generation and saving done.")


@hydra.main(version_base="1.3", config_path="../../configs/datagen", config_name="config.yaml")
def main(cfg: DictConfig):
    """Hydra entry point for label generation."""
    logger.add(cfg.log_file, rotation="10 MB", enqueue=True, backtrace=True, diagnose=True)
    cfg.num_processes, cfg.num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    cfg.max_memory_per_process = configure_max_memory_per_process(
        cfg.get("max_memory_per_process")
    )
    extras(cfg)
    run_label_generation(cfg)


if __name__ == "__main__":
    main()
