"""HORM archive adapter for STRUCTURES25 data generation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
import zarr
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset


class HORM(DataGenDataset):
    """Read HORM-style LMDB archives as STRUCTURES25 datagen inputs.

    The local HORM archives contain pickled ``torch_geometric.data.Data`` objects with at
    least ``charges`` and ``pos``. Archive energy/force/Hessian fields are stored only as
    reference metadata after label generation; the PBE labels come from the STRUCTURES25
    KSDFT and labelgen pipeline.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        archive_file: str,
        name: str = "HORM",
        dataset_name: str | None = None,
        sample_ids: list[int] | None = None,
        charge: int = 0,
        spin: int = 0,
        source_dft_level: str = "omegaB97x/6-31G(d)",
        positions_unit: str = "Angstrom",
        num_processes: int = 1,
    ):
        self.archive_file = archive_file
        self.dataset_name = dataset_name
        self.sample_ids = None if sample_ids is None else np.array(sample_ids, dtype=np.int64)
        self.charge = charge
        self.spin = spin
        self.source_dft_level = source_dft_level
        self.positions_unit = positions_unit
        self._env: lmdb.Environment | None = None
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.archive_path = self.raw_data_dir / self.archive_file
        if not self.archive_path.exists():
            raise FileNotFoundError(f"HORM archive not found: {self.archive_path}")
        if not self.archive_path.is_file():
            raise FileNotFoundError(f"HORM archive path is not a file: {self.archive_path}")
        self.dataset_name = self.dataset_name or self.archive_path.stem
        self.num_archive_entries = self._stat_entries()
        self.num_molecules = self.get_num_molecules()
        logger.info(
            f"Using HORM archive {self.archive_path} with {self.num_archive_entries} entries; "
            f"configured molecule count is {self.num_molecules}."
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def download(self) -> None:
        """HORM archives are expected to be present locally."""
        logger.info("No download configured for HORM; using local archive files.")

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.archive_path.as_posix(),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=1,
            )
        return self._env

    def _stat_entries(self) -> int:
        env = lmdb.open(
            self.archive_path.as_posix(),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1,
        )
        with env.begin(write=False) as txn:
            entries = int(txn.stat()["entries"])
        env.close()
        self._env = None
        return entries

    def _load_raw_sample(self, id: int) -> Any:
        env = self._get_env()
        with env.begin(write=False) as txn:
            value = txn.get(str(int(id)).encode())
        if value is None:
            raise KeyError(f"HORM archive key {id} not found in {self.archive_path}")
        return pickle.loads(value)

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.numpy()
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value

    @staticmethod
    def _replace_dataset(group: zarr.Group, key: str, value: Any, **kwargs) -> None:
        if value is None:
            return
        if key in group:
            del group[key]
        group.create_dataset(key, data=value, **kwargs)

    def get_num_molecules(self) -> int:
        """Get the number of configured molecules."""
        if self.sample_ids is not None:
            return int(len(self.sample_ids))
        return int(self.num_archive_entries)

    def load_charges_and_positions(self, id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load atomic numbers and Angstrom positions for a HORM sample."""
        data = self._load_raw_sample(id)
        charges = self._to_numpy(data.charges).astype(np.uint8)
        positions = self._to_numpy(data.pos).astype(np.float64)
        return charges, positions

    def load_charge_and_spin(self, id: int) -> tuple[int, int]:
        """HORM archives used here do not carry charge/spin; default to neutral singlet."""
        return self.charge, self.spin

    def get_all_atomic_numbers(self) -> np.ndarray:
        """HORM archive datasets contain H/C/N/O."""
        return np.array([1, 6, 7, 8], dtype=np.uint8)

    def get_ids(self) -> np.ndarray:
        """Get archive keys selected for this dataset."""
        if self.sample_ids is not None:
            return np.sort(self.sample_ids)
        return np.arange(self.num_archive_entries, dtype=np.int64)

    def save_reference_metadata(
        self, label_path: Path, molecule_id: int, sample_id: int | None = None
    ) -> None:
        """Append raw HORM reference fields as metadata to a generated zarr label file."""
        data = self._load_raw_sample(molecule_id)
        natoms = int(data.natoms.item()) if hasattr(data.natoms, "item") else int(data.natoms)
        hessian = self._to_numpy(data.hessian).reshape(3 * natoms, 3 * natoms)
        fields = {
            "source_dataset": self.dataset_name,
            "source_path": self.archive_path.as_posix(),
            "source_key": int(molecule_id),
            "source_dft_level": self.source_dft_level,
            "positions_unit": self.positions_unit,
            "energy_unit": "archive_units",
            "forces_unit": "archive_units",
            "hessian_unit": "archive_units",
            "is_training_target_for_pbe_labels": False,
            "natoms": natoms,
            "atomic_numbers": self._to_numpy(data.charges).astype(np.uint8),
            "atom_pos": self._to_numpy(data.pos).astype(np.float64),
            "energy": self._to_numpy(getattr(data, "energy", None)),
            "forces": self._to_numpy(getattr(data, "forces", None)),
            "hessian": hessian,
            "rxn": self._to_numpy(getattr(data, "rxn", None)),
            "ae": self._to_numpy(getattr(data, "ae", None)),
            "freq": self._to_numpy(getattr(data, "freq", None)),
            "eig_values": self._to_numpy(getattr(data, "eig_values", None)),
            "force_constant": self._to_numpy(getattr(data, "force_constant", None)),
        }
        with zarr.ZipStore(label_path, mode="a") as zipstore:
            root = zarr.open(zipstore, mode="a")
            metadata = root.require_group("metadata")
            reference = metadata.require_group("reference")
            for key, value in fields.items():
                self._replace_dataset(reference, key, value, compressor=None)
        logger.info(f"Saved HORM reference metadata for key {molecule_id} to {label_path}")
