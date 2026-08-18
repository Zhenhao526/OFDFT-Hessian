"""PyTorch Lightning DataModule for all molecular datasets."""
import pickle
import platform
from functools import partial
from pathlib import Path

import lightning
import torch
from torch.utils.data.distributed import DistributedSampler

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.pair_batch_sampler import ParentPairBatchSampler
from mldft.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def worker_init_fn(worker_id: int, dtype: torch.dtype):
    """Helper function to set the default torch dtype inside the worker.

    Args:
        worker_id: The worker id, is not used here.
        dtype: The torch dtype to be used inside the data loader.
    """
    torch.set_default_dtype(dtype)


# Optionally, define prepare_data, teardown, test_dataloader
class OFDataModule(lightning.LightningDataModule):
    dataset_class = OFDataset

    def __init__(
        self,
        split_file: Path | str,
        data_dir: Path | str,
        transforms: MasterTransformation,
        basis_info: BasisInfo,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        shuffle_val: bool = True,
        shuffle_test: bool = False,
        use_cached_iterations: bool = True,
        dataset_kwargs: dict = None,
        dataloader_kwargs: dict = None,
        pair_grouped_train_batches: bool = False,
        pair_batch_seed: int = 1729,
        pair_source_markers: list[str] | None = None,
        infer_pair_metadata_from_filename: bool = False,
        pairs_per_batch: int = 1,
        pair_replay_sidecar_dir: str | Path | None = None,
        pair_replay_interval: int = 1,
        pair_replay_phase: int = 0,
        hvp_replay_sidecar_dir: str | Path | None = None,
        hvp_replay_per_batch: bool = False,
        hvp_replay_interval: int = 1,
        hvp_replay_phase: int = 0,
        train_only: bool = False,
    ):
        """Initialize the DataModule.

        Set up the parameters directory, basis info and transforms that will later be used in setup().

        Args:
            split_file: Name of the yaml file containing the train, val, test split
            data_dir: Path to the directory containing the data
            transforms: transforms for data augmentation
            basis_info: BasisInfo object describing which basis functions are used in the dataset
            batch_size: batch size
            num_workers: number of workers for the dataloader
            pin_memory: whether to pin memory
            shuffle_train: whether to shuffle the training set
            shuffle_val: whether to shuffle the validation set
            shuffle_test: whether to shuffle the test set
            use_cached_iterations: whether to load the number of scf iterations from the split file or recompute them
            dataset_kwargs: Keyword arguments passed to :class:`OFDataset`.
            dataloader_kwargs: Keyword arguments passed to :class:`OFLoader`.
            pair_grouped_train_batches: Keep exact paired ground-state geometries in the same
                training mini-batch. Set the trainer's distributed-sampler replacement off when
                this is enabled under DDP.
            pair_batch_seed: Deterministic epoch-shuffle seed for pair-grouped batches.
            pair_source_markers: Optional path components restricting which dataset sources are
                scanned for exact pairs.
            infer_pair_metadata_from_filename: Infer the established ``parent.sample_id`` pair
                convention instead of opening every Zarr file during sampler construction.
            pairs_per_batch: Number of complete displacement pairs placed in each active
                pair-grouped mini-batch.
            train_only: Instantiate only the training split during ``fit`` and reject validation
                or test loader access. This is intended for pre-registered fixed-step training
                protocols that must not read held-out split entries.
        """
        super().__init__()
        self.split_file = Path(split_file)
        self.data_dir = Path(data_dir)
        # Set transforms and configure them
        self.transforms = transforms
        self.label_subdir = transforms.label_subdir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.basis_info = basis_info
        self.use_cached_iterations = use_cached_iterations
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}
        self.pair_grouped_train_batches = bool(pair_grouped_train_batches)
        self.pair_batch_seed = int(pair_batch_seed)
        self.pair_source_markers = list(pair_source_markers or [])
        self.infer_pair_metadata_from_filename = bool(
            infer_pair_metadata_from_filename
        )
        self.pairs_per_batch = int(pairs_per_batch)
        self.pair_replay_sidecar_dir = pair_replay_sidecar_dir
        self.pair_replay_interval = int(pair_replay_interval)
        self.pair_replay_phase = int(pair_replay_phase)
        self.hvp_replay_sidecar_dir = hvp_replay_sidecar_dir
        self.hvp_replay_per_batch = bool(hvp_replay_per_batch)
        self.hvp_replay_interval = int(hvp_replay_interval)
        self.hvp_replay_phase = int(hvp_replay_phase)
        self.train_only = bool(train_only)
        # Get the torch default dtype from the main process and pass it to the workers.
        if platform.system() == "Darwin":
            self.worker_init_fn = partial(worker_init_fn, dtype=torch.get_default_dtype())
            log.warning(
                "Setting worker_init_fn for MacOS. This will break the seeding of the workers."
            )
        else:
            self.worker_init_fn = None

    def setup(self, stage: str):
        """Load the necessary datasets, split into train, validation and test sets.

        Args:
            stage: "fit" (train + validate), "validate" or "test"
        """
        if stage not in ["fit", "validate", "test"]:
            raise ValueError(
                f"Stage '{stage}' not supported. Must be 'fit', 'validate' or 'test'."
            )
        log.info(f"Using labels from {self.label_subdir}.")
        assert self.split_file.exists(), f"Split file {self.split_file} does not exist."
        with self.split_file.open("rb") as f:
            split_dict = pickle.load(f)
        # Load paths and iterations from the split file, split this to avoid unnecessary loading
        if stage == "fit":
            train_paths = [
                self.data_dir / dataset / self.label_subdir / label_path
                for dataset, label_path, _ in split_dict["train"]
            ]
            if self.use_cached_iterations:
                train_iterations = [scf_iterations for _, _, scf_iterations in split_dict["train"]]
            else:
                train_iterations = None
            self.train_set = self.dataset_class(
                paths=train_paths,
                num_scf_iterations_per_path=train_iterations,
                basis_info=self.basis_info,
                transforms=self.transforms,
                **self.dataset_kwargs,
            )
        if stage == "validate" and self.train_only:
            raise RuntimeError("Validation access is disabled in train_only mode")
        if stage == "test" and self.train_only:
            raise RuntimeError("Test access is disabled in train_only mode")
        if (stage == "fit" and not self.train_only) or (
            stage == "validate" and self.val_set is None
        ):
            val_paths = [
                self.data_dir / dataset / self.label_subdir / label_path
                for dataset, label_path, _ in split_dict["val"]
            ]
            if self.use_cached_iterations:
                val_iterations = [scf_iterations for _, _, scf_iterations in split_dict["val"]]
            else:
                val_iterations = None
            self.val_set = self.dataset_class(
                paths=val_paths,
                num_scf_iterations_per_path=val_iterations,
                basis_info=self.basis_info,
                transforms=self.transforms,
                **self.dataset_kwargs,
            )
        elif stage == "test":
            test_paths = [
                self.data_dir / dataset / self.label_subdir / label_path
                for dataset, label_path, _ in split_dict["test"]
            ]
            if self.use_cached_iterations:
                test_iterations = [scf_iterations for _, _, scf_iterations in split_dict["test"]]
            else:
                test_iterations = None
            self.test_set = self.dataset_class(
                paths=test_paths,
                num_scf_iterations_per_path=test_iterations,
                basis_info=self.basis_info,
                transforms=self.transforms,
                **self.dataset_kwargs,
            )

    def train_dataloader(self) -> OFLoader:
        """Return the training dataloader."""
        if self.pair_grouped_train_batches:
            trainer = getattr(self, "trainer", None)
            world_size = int(getattr(trainer, "world_size", 1))
            global_rank = int(getattr(trainer, "global_rank", 0))
            batch_sampler = ParentPairBatchSampler(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                drop_last=True,
                seed=self.pair_batch_seed,
                num_replicas=world_size,
                rank=global_rank,
                pair_source_markers=self.pair_source_markers,
                infer_pair_metadata_from_filename=self.infer_pair_metadata_from_filename,
                pairs_per_batch=self.pairs_per_batch,
                pair_replay_sidecar_dir=self.pair_replay_sidecar_dir,
                pair_replay_interval=self.pair_replay_interval,
                pair_replay_phase=self.pair_replay_phase,
                hvp_replay_sidecar_dir=self.hvp_replay_sidecar_dir,
                hvp_replay_per_batch=self.hvp_replay_per_batch,
                hvp_replay_interval=self.hvp_replay_interval,
                hvp_replay_phase=self.hvp_replay_phase,
            )
            return OFLoader(
                self.train_set,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=self.worker_init_fn,
                **self.dataloader_kwargs,
            )
        return OFLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            drop_last=True,  # Important during training to avoid large gradients
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> OFLoader:
        """Return the validation dataloader."""
        if self.train_only:
            raise RuntimeError("Validation access is disabled in train_only mode")
        sampler = self._distributed_evaluation_sampler(
            self.val_set, shuffle=self.shuffle_val
        )
        return OFLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> OFLoader:
        """Return the test dataloader."""
        if self.train_only:
            raise RuntimeError("Test access is disabled in train_only mode")
        sampler = self._distributed_evaluation_sampler(
            self.test_set, shuffle=self.shuffle_test
        )
        return OFLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.dataloader_kwargs,
        )

    def predict_dataloader(self) -> OFLoader:
        """Return the prediction dataloader."""
        sampler = self._distributed_evaluation_sampler(
            self.predict_set, shuffle=self.shuffle_test
        )
        return OFLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.dataloader_kwargs,
        )

    def _distributed_evaluation_sampler(self, dataset, shuffle: bool):
        """Shard evaluation explicitly when pair batching disables Lightning replacement."""
        if not self.pair_grouped_train_batches:
            return None
        trainer = getattr(self, "trainer", None)
        world_size = int(getattr(trainer, "world_size", 1))
        if world_size <= 1:
            return None
        global_rank = int(getattr(trainer, "global_rank", 0))
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=bool(shuffle),
            seed=self.pair_batch_seed,
            drop_last=False,
        )
