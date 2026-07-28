"""The base class for all datasets.

It implements methods for checking which ids have been computed, which ids still
have to be computed and for verifying the chk files. The following methods have to be implemented by the
the subclasses:

* download(): If the dataset is not yet downloaded, download it.
* get_num_molecules(): Get the number of molecules in the dataset.
* load_charges_and_positions(ids): Load nuclear charges and positions for the given molecule indices.
* get_ids(): Get the indices of the molecules in the dataset.
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
from loguru import logger
from pyscf import gto
from tqdm import tqdm

from mldft.utils.molecules import build_molecule_np


class DataGenDataset(ABC):
    """Base class for all datasets.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the directory containing the raw data.
        kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
        num_processes: Number of processes to use for the computation.
        num_molecules: Number of molecules in the dataset.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str,
        num_processes: int = 1,
    ):
        """Initialize the dataset by setting attributes.

        Args:
            raw_data_dir: Path to the directory containing the raw data.
            kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
        """
        self.filename = filename
        self.name = name
        self.raw_data_dir = Path(raw_data_dir)
        # Can be changed for some datasets
        self.num_processes = num_processes
        self.kohn_sham_data_dir = Path(kohn_sham_data_dir)
        self.kohn_sham_data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.kohn_sham_data_dir.chmod(0o770)
        except PermissionError:
            logger.warning(
                f"Could not set permissions for {self.kohn_sham_data_dir}. This is expected if the directory is not owned by you."
            )
        if not self.raw_data_dir.exists():
            logger.info(f"The configured dataset {name} does not yet exist, downloading it now.")
            try:
                self.download()
            except KeyboardInterrupt:
                logger.warning("Execution interrupted, removing temporary file.")
                shutil.rmtree(self.raw_data_dir)
            except Exception as e:
                logger.warning(f"An error occurred: {e}, removing temporary file.")
                shutil.rmtree(self.raw_data_dir)
                raise e
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.label_dir.chmod(0o770)
        except PermissionError:
            logger.warning(
                f"Could not set permissions for {self.label_dir}. This is expected if the directory is not owned by you."
            )

    @abstractmethod
    def download(self) -> None:
        """Download the raw data."""
        raise NotImplementedError

    @abstractmethod
    def get_num_molecules(self):
        """Get the number of molecules in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def load_charges_and_positions(self, ids: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load nuclear charges and positions for the given molecule indices.

        Args:
            ids: Array of indices of the molecules to compute.
        """
        raise NotImplementedError

    def load_charge_and_spin(self, id: int) -> tuple[int | None, int | None]:
        """Load total molecular charge and PySCF spin for a molecule.

        PySCF defines spin as N_alpha - N_beta, so singlet closed-shell molecules have spin 0.
        Existing datasets do not carry charge/spin annotations and keep PySCF defaults.
        """
        return None, None

    def get_sample_ids(self, id: int) -> tuple[int | None, ...]:
        """Return geometry/sample ids that should be generated for one molecule id.

        Datasets without per-molecule sampling keep the historical filename without a sample suffix
        by returning ``(None,)``. Sampled datasets should return integer sample ids.
        """
        return (None,)

    def load_sample(
        self, id: int, sample_id: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
        """Load charges, positions, charge and spin for a molecule sample."""
        charges, positions = self.load_charges_and_positions(id)
        charge, spin = self.load_charge_and_spin(id)
        return charges, positions, charge, spin

    def load_molecule(self, id: int, basis: str) -> gto.Mole:
        """Load nuclear charges and positions for the given molecule index.

        Args:
            id: Index of the molecule to compute.
            basis: Basis set to use for the molecule.
        """
        charges, positions, charge, spin = self.load_sample(id)
        return build_molecule_np(
            charges, positions, basis=basis, unit="Angstrom", charge=charge, spin=spin
        )

    def save_reference_metadata(
        self, label_path: Path, molecule_id: int, sample_id: int | None = None
    ) -> None:
        """Optionally append dataset-specific reference metadata to a generated label file."""
        return None

    def get_all_atomic_numbers(self) -> np.ndarray:
        """Get the atomic numbers of all atoms in the dataset.

        Returns:
            np.ndarray: Array of atomic numbers.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset."""
        raise NotImplementedError

    def get_ids_done_ks(self) -> np.ndarray:
        """Get the indices of the molecules that have already been computed.

        Returns:
            np.ndarray: Array of indices of the molecules that have already been computed.
        """
        return self.get_ids_done_ks_for_ids(self.get_ids())

    def get_ids_done_ks_for_ids(self, ids: np.ndarray) -> np.ndarray:
        """Get completed molecule ids, restricted to the provided ids."""
        return np.array(
            [
                int(id)
                for id in ids
                if all(
                    self.get_chk_file_from_id(int(id), sample_id).exists()
                    for sample_id in self.get_sample_ids(int(id))
                )
            ],
            dtype=int,
        )

    def get_label_file_from_id(self, id: int, sample_id: int | None = None) -> Path:
        """Get the label path for a molecule id/sample id pair."""
        if sample_id is None:
            return self.label_dir / f"{id:07}.zarr.zip"
        return self.label_dir / f"{id:07}.{sample_id:07}.zarr.zip"

    def get_ids_done_labelgen(self) -> np.ndarray:
        """Get the indices of the molecules whose labels have already been computed.

        Returns:
            np.ndarray: Array of indices of the molecules that have already been computed.
        """
        return self.get_ids_done_labelgen_for_ids(self.get_ids())

    def get_ids_done_labelgen_for_ids(self, ids: np.ndarray) -> np.ndarray:
        """Get completed label ids, restricted to the provided ids."""
        return np.array(
            [
                int(id)
                for id in ids
                if all(
                    self.get_label_file_from_id(int(id), sample_id).exists()
                    for sample_id in self.get_sample_ids(int(id))
                )
            ],
            dtype=int,
        )

    def get_ids_todo_ks(self, start_idx: int = 0, max_num_molecules: int = 1) -> np.ndarray:
        """Get the indices of the molecules that haven't been computed, typically by comparing
        total indices with indices of already computed molecules.

        Args:
            start_idx: Index of the first molecule to compute.
            max_num_molecules: Number of molecules to compute.

        Returns:
            np.ndarray: Array of indices of the molecules that haven't been computed.
        """
        end_idx = (
            start_idx + max_num_molecules if max_num_molecules > 0 else self.get_num_molecules()
        )
        ids = self.get_ids()[start_idx:end_idx]
        ids_done = self.get_ids_done_ks_for_ids(ids)
        ids_todo = np.setdiff1d(ids, ids_done)
        logger.info(
            f"Configured to run {max_num_molecules} molecules starting at index {start_idx}, found {len(ids) - len(ids_todo)} already done, processing {len(ids_todo)}."
        )
        return ids_todo

    def get_ids_todo_labelgen(self, start_idx: int = 0, max_num_molecules: int = 1) -> np.ndarray:
        """Get the indices of the molecules that haven't been computed, typically by comparing
        total indices with indices of already computed molecules.

        Args:
            start_idx: Index of the first molecule to compute.
            max_num_molecules: Number of molecules to compute.

        Returns:
            np.ndarray: Array of indices of the molecules that haven't been computed.
        """
        end_idx = (
            start_idx + max_num_molecules if max_num_molecules > 0 else self.get_num_molecules()
        )
        ids = self.get_ids()[start_idx:end_idx]
        ids_done = self.get_ids_done_labelgen_for_ids(ids)
        ids_todo = np.setdiff1d(ids, ids_done)
        logger.info(
            f"Configured to run {max_num_molecules} molecules starting at index {start_idx}, found {len(ids) - len(ids_todo)} already done, processing {len(ids_todo)}."
        )
        return ids_todo

    def get_chk_file_from_id(self, id: int, sample_id: int | None = None) -> Path:
        """Get the path to the chk file for the given molecule index/sample id.

        Args:
            id: Index of the molecule to compute.
            sample_id: Optional sample id for datasets with multiple geometries per molecule.

        Returns:
            Path: Path to the chk file.
        """
        if sample_id is None:
            return self.kohn_sham_data_dir / f"{self.filename}_{id:07}.chk"
        return self.kohn_sham_data_dir / f"{self.filename}_{id:07}.{sample_id:07}.chk"

    def get_all_chk_files_from_id(self, id: int) -> Sequence[Path]:
        """Get the paths to all possible chk files from an id, including those from external
        potential sampling.

        Returns:
            Sequence[Path]: Array of paths to the chk files.
        """
        return list(self.kohn_sham_data_dir.glob(f"{self.filename}_{id:07}*.chk"))

    def get_all_chk_files_from_ids(self, ids: np.ndarray) -> Sequence[Path]:
        """Get the paths to all possible chk files from a list of id, including those from external
        potential sampling.

        Returns:
            Sequence[Path]: Array of paths to the chk files.
        """
        return [
            file
            for file in sorted(self.kohn_sham_data_dir.glob(f"{self.filename}_*.chk"))
            if int(file.stem.split("_")[-1].split(".")[0]) in ids
        ]

    def verify_files(self, remove_broken_files: bool = True) -> None:
        """Remove the files that are not finished.

        Args:
            remove_broken_files: Whether to remove the broken files or raise an error.
        """
        logger.info(f"Verifying chk files in {self.kohn_sham_data_dir}")
        files = list(self.kohn_sham_data_dir.glob("*.chk"))
        for file in tqdm(files, position=0, leave=False, dynamic_ncols=True):
            self.check_chk_file(file, remove_broken_files=remove_broken_files)

    @staticmethod
    def check_chk_file(chk_file: Path, remove_broken_files: bool = True) -> None:
        """Check if the computation of the chk file is finished and remove it if it is not.

        Args:
            chk_file: Path to the chk file.
            remove_broken_files: Whether to remove the broken files or raise an error.
        """
        try:
            with h5py.File(chk_file, "r") as f:
                if "Results" not in f:
                    logger.info(f"Removing {chk_file}.")
                    if remove_broken_files:
                        chk_file.unlink()
                    else:
                        raise RuntimeError(f"Found broken file {chk_file}.")
        except OSError:
            logger.warning(f"Found broken file resulting in OS error, removing it {chk_file}.")
            if remove_broken_files:
                chk_file.unlink()
            else:
                raise RuntimeError(f"Found broken file {chk_file}.")


def delete_dataset(dataset: DataGenDataset) -> None:
    """Delete the dataset.

    Deletes the raw data, koohn-sham data and label directories.

    Args:
        dataset: The dataset to delete.
    """
    shutil.rmtree(dataset.raw_data_dir)
    shutil.rmtree(dataset.kohn_sham_data_dir)
    for dir in dataset.label_dir.parent.glob("label*"):
        shutil.rmtree(dir)
