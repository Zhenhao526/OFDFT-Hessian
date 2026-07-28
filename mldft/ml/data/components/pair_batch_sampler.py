"""Parent-safe batch sampler that keeps exact geometry pairs in the same mini-batch."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from math import lcm
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Sampler


class ParentPairBatchSampler(Sampler[list[int]]):
    """Batch exact ``R+``/``R-`` ground-state samples without splitting a pair across ranks."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 1729,
        num_replicas: int | None = None,
        rank: int | None = None,
        pair_source_markers: list[str] | tuple[str, ...] | None = None,
        infer_pair_metadata_from_filename: bool = False,
        pair_replay_sidecar_dir: str | Path | None = None,
        pair_replay_interval: int = 1,
        pair_replay_phase: int = 0,
        hvp_replay_sidecar_dir: str | Path | None = None,
        hvp_replay_per_batch: bool = False,
        hvp_replay_interval: int = 1,
        hvp_replay_phase: int = 0,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Pair-grouped batches require batch_size >= 2")
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.pair_source_markers = tuple(pair_source_markers or ())
        self.infer_pair_metadata_from_filename = bool(
            infer_pair_metadata_from_filename
        )
        self.pair_replay_sidecar_dir = (
            None if pair_replay_sidecar_dir is None else Path(pair_replay_sidecar_dir)
        )
        if pair_replay_interval <= 0 or not 0 <= pair_replay_phase < pair_replay_interval:
            raise ValueError("Invalid pair replay interval/phase")
        if hvp_replay_interval <= 0 or not 0 <= hvp_replay_phase < hvp_replay_interval:
            raise ValueError("Invalid HVP replay interval/phase")
        self.pair_replay_interval = int(pair_replay_interval)
        self.pair_replay_phase = int(pair_replay_phase)
        self.hvp_replay_sidecar_dir = (
            None if hvp_replay_sidecar_dir is None else Path(hvp_replay_sidecar_dir)
        )
        self.hvp_replay_per_batch = bool(hvp_replay_per_batch)
        self.hvp_replay_interval = int(hvp_replay_interval)
        self.hvp_replay_phase = int(hvp_replay_phase)
        self.epoch = 0
        self.pairs, self.other_indices = self._index_dataset()
        if not self.pairs:
            raise ValueError("No complete exact geometry pairs found in the training dataset")
        self.hvp_replay_indices = self._index_hvp_replay()
        if self.hvp_replay_per_batch:
            if not self.hvp_replay_indices:
                raise ValueError("No stable HVP replay samples found")
            hvp_set = set(self.hvp_replay_indices)
            self.other_indices = [
                index for index in self.other_indices if index not in hvp_set
            ]

    @staticmethod
    def _strictly_stable_parents(sidecar_dir: Path) -> set[int]:
        stable_parents = set()
        for sidecar in sorted(sidecar_dir.glob("*.0000000.npz")):
            payload = np.load(sidecar)
            mask = np.asarray(
                payload.get("stability_mask", np.asarray([True])), dtype=np.bool_
            ).reshape(-1)
            explicitly_eligible = payload.get("parent_stability_eligible")
            eligible = (
                bool(np.asarray(explicitly_eligible).reshape(()))
                if explicitly_eligible is not None
                else bool(mask.all())
            )
            if mask.size and bool(mask.any()) and eligible:
                stable_parents.add(int(sidecar.name.split(".")[0]))
        return stable_parents

    @property
    def sampler(self) -> "ParentPairBatchSampler":
        """Expose the epoch-aware sampler interface expected by Lightning.

        Lightning calls ``set_epoch`` on ``dataloader.batch_sampler.sampler``. This sampler
        already owns complete batches, so returning itself is sufficient and keeps pair/rank
        sharding deterministic while changing the order between epochs.
        """
        return self

    def _index_dataset(self) -> tuple[list[tuple[int, int]], list[int]]:
        grouped: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        paired_path_indices: set[int] = set()
        pair_parents = (
            None
            if self.pair_replay_sidecar_dir is None
            else self._strictly_stable_parents(self.pair_replay_sidecar_dir)
        )
        for path_index, path in enumerate(self.dataset.paths):
            if self.pair_source_markers and not any(
                marker in path.parts for marker in self.pair_source_markers
            ):
                continue
            if self.infer_pair_metadata_from_filename:
                parts = path.name.split(".")
                if len(parts) < 2:
                    raise ValueError(f"Cannot parse paired label filename {path.name}")
                source = int(parts[0])
                sample_id = int(parts[1])
                pair_id = (sample_id + 1) // 2
                sign = 1 if sample_id % 2 == 1 else -1
            else:
                root = zarr.open(path, mode="r")
                if "metadata/reference" not in root:
                    continue
                reference = root["metadata/reference"]
                if "paired_perturbations" not in reference or not bool(
                    reference["paired_perturbations"][()]
                ):
                    continue
                sign = int(reference["perturbation_pair_sign"][()])
                if sign not in (-1, 1):
                    continue
                source = int(reference["source_molecule_id"][()])
                pair_id = int(reference["perturbation_pair_id"][()])
            scf_iterations = np.asarray(
                self.dataset.scf_iterations_per_path[path_index], dtype=np.int64
            )
            if scf_iterations.size == 0:
                continue
            path_start = int(self.dataset.path_indices[path_index])
            path_stop = int(self.dataset.path_indices[path_index + 1])
            paired_path_indices.update(range(path_start, path_stop))
            if pair_parents is not None and source not in pair_parents:
                continue
            ground_state = int(np.max(scf_iterations))
            local = np.flatnonzero(scf_iterations == ground_state)
            if local.size != 1:
                raise ValueError(f"Could not identify one ground-state sample for {path}")
            index = int(self.dataset.path_indices[path_index] + local[0])
            key = (source, pair_id)
            if sign in grouped[key]:
                raise ValueError(f"Duplicate sign {sign} for pair {key}")
            grouped[key][sign] = index

        incomplete = [key for key, values in grouped.items() if set(values) != {-1, 1}]
        if incomplete:
            raise ValueError(f"Incomplete exact geometry pairs: {incomplete[:10]}")
        pairs = []
        for key in sorted(grouped):
            pair = (grouped[key][-1], grouped[key][1])
            pairs.append(pair)
        # Pair metadata is geometry-level metadata and is therefore attached to every SCF
        # iteration from a paired file. Only the ground-state R-/R+ samples form the exact energy
        # secant; exclude all other iterations from those files so they cannot enter a batch as an
        # apparently incomplete pair.
        other_indices = [
            index for index in range(len(self.dataset)) if index not in paired_path_indices
        ]
        return pairs, other_indices

    def _index_hvp_replay(self) -> list[int]:
        if not self.hvp_replay_per_batch:
            return []
        if self.hvp_replay_sidecar_dir is None:
            raise ValueError("hvp_replay_per_batch=True requires hvp_replay_sidecar_dir")
        stable_parents = self._strictly_stable_parents(self.hvp_replay_sidecar_dir)
        indices = []
        for path_index, path in enumerate(self.dataset.paths):
            parts = path.name.removesuffix(".zarr.zip").split(".")
            if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            source, sample_id = int(parts[0]), int(parts[1])
            if source not in stable_parents or sample_id != 0:
                continue
            scf_iterations = np.asarray(
                self.dataset.scf_iterations_per_path[path_index], dtype=np.int64
            )
            ground_state = int(np.max(scf_iterations))
            local = np.flatnonzero(scf_iterations == ground_state)
            if local.size != 1:
                raise ValueError(f"Could not identify one HVP ground state for {path}")
            indices.append(int(self.dataset.path_indices[path_index] + local[0]))
        return sorted(indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        set_dataset_epoch = getattr(self.dataset, "set_epoch", None)
        if set_dataset_epoch is not None:
            set_dataset_epoch(self.epoch)

    def _global_batches(self) -> list[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        pair_order = np.arange(len(self.pairs))
        other_indices = np.asarray(self.other_indices, dtype=np.int64)
        hvp_indices = np.asarray(self.hvp_replay_indices, dtype=np.int64)
        if self.shuffle:
            generator.shuffle(pair_order)
            generator.shuffle(other_indices)
            generator.shuffle(hvp_indices)

        def pair_active(batch_index: int) -> bool:
            return batch_index % self.pair_replay_interval == self.pair_replay_phase

        def hvp_active(batch_index: int) -> bool:
            return self.hvp_replay_per_batch and (
                batch_index % self.hvp_replay_interval == self.hvp_replay_phase
            )

        period = lcm(
            self.pair_replay_interval,
            self.hvp_replay_interval if self.hvp_replay_per_batch else 1,
        )
        fill_pattern = [
            self.batch_size
            - (2 if pair_active(index) else 0)
            - (1 if hvp_active(index) else 0)
            for index in range(period)
        ]
        if min(fill_pattern) < 0:
            raise ValueError("Replay schedule requests more special graphs than batch_size")
        can_replay_ordinary = max(fill_pattern) > 0
        if can_replay_ordinary and len(other_indices) == 0:
            raise ValueError("Replay schedule requires ordinary energy/force samples")

        n_batches = 0
        ordinary_capacity = 0
        pair_batches = 0
        hvp_batches = 0
        while (
            pair_batches < len(pair_order)
            or (self.hvp_replay_per_batch and hvp_batches < len(hvp_indices))
            or (can_replay_ordinary and ordinary_capacity < len(other_indices))
        ):
            pair_batches += int(pair_active(n_batches))
            hvp_batches += int(hvp_active(n_batches))
            ordinary_capacity += fill_pattern[n_batches % period]
            n_batches += 1

        batches: list[list[int]] = []
        pair_cursor = 0
        hvp_cursor = 0
        ordinary_cursor = 0
        for batch_index in range(n_batches):
            batch = []
            if pair_active(batch_index):
                pair = self.pairs[int(pair_order[pair_cursor % len(pair_order)])]
                batch.extend(pair)
                pair_cursor += 1
            if hvp_active(batch_index):
                batch.append(int(hvp_indices[hvp_cursor % len(hvp_indices)]))
                hvp_cursor += 1
            fill_count = fill_pattern[batch_index % period]
            for offset in range(fill_count):
                other_index = (ordinary_cursor + offset) % len(other_indices)
                batch.append(int(other_indices[other_index]))
            ordinary_cursor += fill_count
            batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)

        remainder = len(batches) % self.num_replicas
        if remainder:
            batches.extend(batches[: self.num_replicas - remainder])
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches()
        yield from batches[self.rank :: self.num_replicas]

    def __len__(self) -> int:
        return len(self._global_batches()) // self.num_replicas
