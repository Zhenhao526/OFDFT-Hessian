"""QM9 dataset.

Contains 129,133 molecules from the QM9 dataset. The ids of the molecules are given by the index of
the xyz file.
"""

import multiprocessing
from pathlib import Path

import numpy as np
import zarr
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.download import download_file, extract_tar
from mldft.utils.molecules import read_xyz_file


class QM9(DataGenDataset):
    """Class for the QM9 dataset.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QM9",
        num_processes: int = 1,
    ):
        """Initialize the QM9 dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.

        Raises:
            AssertionError: If the subset is not in the list of available subsets.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.filename = filename.split(".")[0]
        self.num_molecules = self.get_num_molecules()

    def download(self) -> None:
        """Download the raw data."""
        logger.info("Downloading QM9")
        downloaded_file = download_file(
            "https://ndownloader.figshare.com/files/3195389",
            self.raw_data_dir,
            filename="dsgdb9nsd.xyz.tar.bz2",
        )
        logger.info("Extracting QM9")
        extract_tar(downloaded_file, self.raw_data_dir, mode="r:bz2")
        downloaded_file.unlink()
        self.convert_xyz_files()

    def convert_xyz_files(self) -> None:
        """Convert the xyz files from QM9 to have the format 1e-6 instead of 1*^-6 which can't be
        read by pyscf."""
        logger.info(f"Converting xyz files to {self.raw_data_dir}")
        convert_folder_sorted_parallel(self.raw_data_dir, self.raw_data_dir, self.num_processes)

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return len(list(self.raw_data_dir.glob("*.xyz")))

    def get_all_atomic_numbers(self) -> np.ndarray:
        return np.array([1, 6, 7, 8, 9])

    def load_charges_and_positions(self, id: int) -> tuple[list, list]:
        """Load nuclear charges and positions for the given molecule indices from the .xyz files.
        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (A).
            np.ndarray: Array of atomic positions (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        file_name = self.raw_data_dir / f"dsgdb9nsd_{id:06}.xyz"
        charges, positions = read_xyz_file(file_name)
        return charges, positions

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset.

        Returns:
            np.ndarray: Array of indices of the molecules in the dataset.
        """
        if (
            self.num_molecules > 0
            and (self.raw_data_dir / "dsgdb9nsd_000001.xyz").exists()
            and (self.raw_data_dir / f"dsgdb9nsd_{self.num_molecules:06}.xyz").exists()
        ):
            return np.arange(1, self.num_molecules + 1, dtype=np.int64)
        return np.sort(
            np.array(
                [int(f.stem.split("_")[1]) for f in self.raw_data_dir.glob("*.xyz")]
            )
        )


def convert_string_format(xyz_file_path: Path, out_folder: Path) -> None:
    """Convert the xyz files from QM9 to have the format 1e-6 instead of 1*^-6 which can't be read
    by pyscf.

    Args:
        xyz_file_path: Path to the xyz file
        out_folder: Path to the output folder
    """
    with open(xyz_file_path) as xyz_file:
        xyz_content = xyz_file.read()
    xyz_content = xyz_content.replace("*^", "e")
    with open(out_folder / xyz_file_path.name, "w") as xyz_file:
        xyz_file.write(xyz_content)


def convert_folder_sorted_parallel(in_folder: Path, out_folder: Path, num_processes: int) -> None:
    """Apply the conversion function to all xyz files in the folder in parallel.

    Args:
        in_folder: Path to the input folder
        out_folder: Path to the output folder
        num_processes: Number of processes to use
    """
    # Get the xyz files in the folder, sort and combine with the output folder
    out_folder.mkdir(exist_ok=True)
    out_folder.chmod(0o770)
    file_out_folder = [
        (xyz_file_path, out_folder) for xyz_file_path in sorted(in_folder.glob("*.xyz"))
    ]
    # Multiprocess for faster conversion
    logger.info(f"Using {num_processes} processes")
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(convert_string_format, file_out_folder)


class QM9Test(QM9):
    def download(self) -> None:
        downloaded_file = download_file(
            "https://figshare.com/ndownloader/files/3195398",
            self.raw_data_dir,
            filename="dsC7O2H10nsd.xyz.tar.bz2",
        )
        logger.info("Extracting QM9")
        extract_tar(downloaded_file, self.raw_data_dir, mode="r:bz2")
        downloaded_file.unlink()
        self.convert_xyz_files()

    def load_charges_and_positions(self, id: int) -> tuple[list, list]:
        """Load nuclear charges and positions for the given molecule indices from the .xyz files.
        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (A).
            np.ndarray: Array of atomic positions (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        file_name = self.raw_data_dir / f"dsC7O2H10nsd_{id:04}.xyz"
        charges, positions = read_xyz_file(file_name)
        return charges, positions


class QM9GeometryPerturbed(QM9):
    """QM9 dataset with deterministic nuclear-coordinate perturbation samples.

    Sample id ``0`` is the unperturbed QM9 geometry. Sample ids ``1..N`` are Gaussian
    coordinate perturbations in Angstrom.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QM9GeometryPerturbed",
        num_perturbations: int = 1,
        perturbation_std: float = 0.01,
        perturbation_distribution: str = "gaussian",
        max_displacement: float | None = 0.05,
        random_seed: int = 20260701,
        include_reference: bool = True,
        remove_translation: bool = True,
        paired_perturbations: bool = False,
        num_processes: int = 1,
    ):
        self.num_perturbations = int(num_perturbations)
        self.perturbation_std = float(perturbation_std)
        self.perturbation_distribution = str(perturbation_distribution).lower()
        self.max_displacement = None if max_displacement is None else float(max_displacement)
        self.random_seed = int(random_seed)
        self.include_reference = bool(include_reference)
        self.remove_translation = bool(remove_translation)
        self.paired_perturbations = bool(paired_perturbations)
        if self.num_perturbations < 0:
            raise ValueError("num_perturbations must be non-negative.")
        if self.perturbation_std < 0:
            raise ValueError("perturbation_std must be non-negative.")
        if self.perturbation_distribution not in {"gaussian", "rademacher"}:
            raise ValueError(
                "perturbation_distribution must be 'gaussian' or 'rademacher'."
            )
        if self.paired_perturbations and self.num_perturbations % 2 != 0:
            raise ValueError("paired_perturbations requires an even num_perturbations.")
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )

    def get_sample_ids(self, id: int) -> tuple[int, ...]:
        sample_ids = []
        if self.include_reference:
            sample_ids.append(0)
        sample_ids.extend(range(1, self.num_perturbations + 1))
        return tuple(sample_ids)

    def _rng_for_sample(self, id: int, sample_id: int) -> np.random.Generator:
        perturbation_id = (int(sample_id) + 1) // 2 if self.paired_perturbations else int(sample_id)
        seed = self.random_seed + int(id) * 1_000_003 + perturbation_id * 97_919
        return np.random.default_rng(seed)

    def _perturb_positions(
        self, positions: np.ndarray, id: int, sample_id: int | None
    ) -> np.ndarray:
        if sample_id in (None, 0):
            return positions.copy()
        rng = self._rng_for_sample(id, sample_id)
        if self.perturbation_distribution == "gaussian":
            displacement = rng.normal(
                loc=0.0, scale=self.perturbation_std, size=positions.shape
            )
        else:
            displacement = (
                rng.choice(np.asarray([-1.0, 1.0]), size=positions.shape)
                * self.perturbation_std
            )
        if self.remove_translation:
            displacement = displacement - displacement.mean(axis=0, keepdims=True)
        if self.max_displacement is not None:
            norms = np.linalg.norm(displacement, axis=1)
            scale = np.ones_like(norms)
            nonzero = norms > 0
            scale[nonzero] = np.minimum(1.0, self.max_displacement / norms[nonzero])
            displacement = displacement * scale[:, None]
        if self.paired_perturbations and int(sample_id) % 2 == 0:
            displacement = -displacement
        return positions + displacement

    def load_sample(
        self, id: int, sample_id: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
        if sample_id is None:
            sample_id = 0 if self.include_reference else 1
        if sample_id not in self.get_sample_ids(id):
            raise ValueError(f"Unknown QM9 geometry sample id {sample_id} for molecule {id}.")
        charges, positions = self.load_charges_and_positions(id)
        positions = self._perturb_positions(positions, id, sample_id)
        charge, spin = self.load_charge_and_spin(id)
        return charges, positions, charge, spin

    @staticmethod
    def _replace_dataset(group: zarr.Group, key: str, value) -> None:
        if key in group:
            del group[key]
        group.create_dataset(key, data=value, compressor=None)

    def save_reference_metadata(
        self, label_path: Path, molecule_id: int, sample_id: int | None = None
    ) -> None:
        charges, reference_positions = self.load_charges_and_positions(molecule_id)
        _, sample_positions, charge, spin = self.load_sample(molecule_id, sample_id)
        fields = {
            "source_dataset": "QM9",
            "source_molecule_id": int(molecule_id),
            "source_filename": f"dsgdb9nsd_{molecule_id:06}.xyz",
            "sample_id": -1 if sample_id is None else int(sample_id),
            "is_reference_geometry": bool(sample_id in (None, 0)),
            "positions_unit": "Angstrom",
            "atomic_numbers": charges.astype(np.uint8),
            "reference_atom_pos": reference_positions.astype(np.float64),
            "sample_atom_pos": sample_positions.astype(np.float64),
            "charge": -999 if charge is None else int(charge),
            "spin": -999 if spin is None else int(spin),
            "num_perturbations": int(self.num_perturbations),
            "perturbation_std_angstrom": float(
                0.0 if sample_id in (None, 0) else self.perturbation_std
            ),
            "perturbation_distribution": self.perturbation_distribution,
            "max_displacement_angstrom": float(
                -1.0 if self.max_displacement is None else self.max_displacement
            ),
            "random_seed": int(self.random_seed),
            "remove_translation": bool(self.remove_translation),
            "paired_perturbations": bool(self.paired_perturbations),
            "perturbation_pair_id": int(
                -1 if sample_id in (None, 0) else (int(sample_id) + 1) // 2
            ),
            "perturbation_pair_sign": int(
                0
                if sample_id in (None, 0) or not self.paired_perturbations
                else (1 if int(sample_id) % 2 == 1 else -1)
            ),
        }
        with zarr.ZipStore(label_path, mode="a") as zipstore:
            root = zarr.open(zipstore, mode="a")
            metadata = root.require_group("metadata")
            reference = metadata.require_group("reference")
            for key, value in fields.items():
                self._replace_dataset(reference, key, value)
        logger.info(
            f"Saved QM9 geometry perturbation metadata for molecule {molecule_id}, "
            f"sample {sample_id} to {label_path}"
        )
