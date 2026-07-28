from pathlib import Path

import numpy as np
import zarr

from mldft.ml.data.components.pair_batch_sampler import ParentPairBatchSampler


class _Dataset:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.scf_iterations_per_path = [np.asarray([0, 1, 2]) for _ in paths]
        self.path_indices = np.arange(0, 3 * (len(paths) + 1), 3)

    def __len__(self):
        return int(self.path_indices[-1])

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def _write_reference(path: Path, source: int, pair_id: int, sign: int) -> None:
    root = zarr.open(path, mode="w")
    reference = root.create_group("metadata").create_group("reference")
    reference.create_dataset("paired_perturbations", data=np.asarray(True))
    reference.create_dataset("source_molecule_id", data=np.asarray(source))
    reference.create_dataset("perturbation_pair_id", data=np.asarray(pair_id))
    reference.create_dataset("perturbation_pair_sign", data=np.asarray(sign))


def _add_ordinary_paths(tmp_path: Path, paths: list[Path], count: int = 4) -> None:
    for index in range(count):
        path = tmp_path / f"ordinary_{index}.zarr"
        zarr.open(path, mode="w")
        paths.append(path)


def test_parent_pair_batch_sampler_keeps_ground_state_pairs_together(tmp_path):
    paths = []
    expected_pairs = []
    for source in range(4):
        pair = []
        for sign in (-1, 1):
            path = tmp_path / f"{source}_{sign}.zarr"
            _write_reference(path, source=source, pair_id=1, sign=sign)
            paths.append(path)
            pair.append(3 * (len(paths) - 1) + 2)
        expected_pairs.append(set(pair))
    paired_path_count = len(paths)
    _add_ordinary_paths(tmp_path, paths)
    dataset = _Dataset(paths)
    sampler = ParentPairBatchSampler(
        dataset, batch_size=4, shuffle=True, seed=17, num_replicas=1, rank=0
    )
    batches = [set(batch) for batch in sampler]

    for pair in expected_pairs:
        containing = [batch for batch in batches if batch & pair]
        assert containing
        assert all(pair <= batch for batch in containing)
    assert all(any(pair <= batch for pair in expected_pairs) for batch in batches)
    paired_non_ground = {
        3 * path_index + scf_iteration
        for path_index in range(paired_path_count)
        for scf_iteration in (0, 1)
    }
    assert all(not (batch & paired_non_ground) for batch in batches)


def test_parent_pair_batch_sampler_shards_whole_batches_across_ranks(tmp_path):
    paths = []
    for source in range(8):
        for sign in (-1, 1):
            path = tmp_path / f"{source}_{sign}.zarr"
            _write_reference(path, source=source, pair_id=1, sign=sign)
            paths.append(path)
    _add_ordinary_paths(tmp_path, paths, count=8)
    dataset = _Dataset(paths)
    rank_batches = []
    for rank in (0, 1):
        sampler = ParentPairBatchSampler(
            dataset,
            batch_size=4,
            shuffle=False,
            num_replicas=2,
            rank=rank,
        )
        rank_batches.append([tuple(batch) for batch in sampler])

    assert len(rank_batches[0]) == len(rank_batches[1])
    assert set(rank_batches[0]).isdisjoint(rank_batches[1])


def test_parent_pair_batch_sampler_can_infer_verified_filename_convention(tmp_path):
    source = tmp_path / "QM9PBEForceRandom1000PairedTrain"
    paths = [
        source / "0000042.0000001.zarr.zip",
        source / "0000042.0000002.zarr.zip",
    ]
    dataset = _Dataset(paths)
    sampler = ParentPairBatchSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        pair_source_markers=["QM9PBEForceRandom1000PairedTrain"],
        infer_pair_metadata_from_filename=True,
    )

    assert [set(batch) for batch in sampler] == [{2, 5}]


def test_parent_pair_batch_sampler_exposes_epoch_aware_sampler(tmp_path):
    paths = []
    for source in range(6):
        for sign in (-1, 1):
            path = tmp_path / f"{source}_{sign}.zarr"
            _write_reference(path, source=source, pair_id=1, sign=sign)
            paths.append(path)
    _add_ordinary_paths(tmp_path, paths, count=6)
    dataset = _Dataset(paths)
    sampler = ParentPairBatchSampler(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=31,
        num_replicas=1,
        rank=0,
    )

    epoch_zero = list(sampler)
    sampler.sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert sampler.sampler is sampler
    assert dataset.epoch == 1
    assert epoch_zero != epoch_one


def test_parent_pair_batch_sampler_replays_one_stable_hvp_parent(tmp_path):
    paired_source = tmp_path / "QM9PBEForceRandom1000PairedTrain"
    paths = []
    for source in range(3):
        for sample_id in (1, 2):
            paths.append(paired_source / f"{source:07d}.{sample_id:07d}.zarr.zip")
    base_source = tmp_path / "QM9PBEForceRandom1000"
    for source in range(3):
        paths.append(base_source / f"{source:07d}.0000000.zarr.zip")
    ordinary_start = len(paths)
    _add_ordinary_paths(tmp_path, paths, count=6)
    dataset = _Dataset(paths)
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    np.savez_compressed(
        sidecars / "0000000.0000000.npz",
        stability_mask=np.asarray([True, True]),
    )
    np.savez_compressed(
        sidecars / "0000001.0000000.npz",
        stability_mask=np.asarray([False, False]),
    )
    stable_ground_state = 3 * (6 + 0) + 2
    unstable_ground_state = 3 * (6 + 1) + 2
    sampler = ParentPairBatchSampler(
        dataset,
        batch_size=4,
        shuffle=False,
        pair_source_markers=["QM9PBEForceRandom1000PairedTrain"],
        infer_pair_metadata_from_filename=True,
        hvp_replay_sidecar_dir=sidecars,
        hvp_replay_per_batch=True,
    )

    batches = list(sampler)
    assert batches
    assert sampler.hvp_replay_indices == [stable_ground_state]
    assert all(stable_ground_state in batch for batch in batches)
    # The excluded parent may still enter ordinary energy/force replay, but never HVP replay.
    assert unstable_ground_state not in sampler.hvp_replay_indices
    assert any(any(index >= 3 * ordinary_start for index in batch) for batch in batches)


def test_parent_pair_batch_sampler_uses_explicit_hierarchical_parent_eligibility(
    tmp_path,
):
    paired_source = tmp_path / "QM9PBEForceRandom1000PairedTrain"
    paths = [
        paired_source / f"{source:07d}.{sample_id:07d}.zarr.zip"
        for source in range(2)
        for sample_id in (1, 2)
    ]
    _add_ordinary_paths(tmp_path, paths, count=4)
    dataset = _Dataset(paths)
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    np.savez_compressed(
        sidecars / "0000000.0000000.npz",
        stability_mask=np.asarray([True, True]),
    )
    np.savez_compressed(
        sidecars / "0000001.0000000.npz",
        stability_mask=np.asarray([True, False]),
        parent_stability_eligible=np.asarray(True),
    )
    sampler = ParentPairBatchSampler(
        dataset,
        batch_size=4,
        shuffle=False,
        pair_source_markers=["QM9PBEForceRandom1000PairedTrain"],
        infer_pair_metadata_from_filename=True,
        pair_replay_sidecar_dir=sidecars,
    )

    assert sampler.pairs == [(5, 2), (11, 8)]


def test_parent_pair_batch_sampler_interleaves_pair_hvp_and_ordinary_replay(tmp_path):
    paired_source = tmp_path / "QM9PBEForceRandom1000PairedTrain"
    paths = [
        paired_source / f"0000000.{sample_id:07d}.zarr.zip"
        for sample_id in (1, 2)
    ]
    base_source = tmp_path / "QM9PBEForceRandom1000"
    paths.append(base_source / "0000000.0000000.zarr.zip")
    _add_ordinary_paths(tmp_path, paths, count=12)
    dataset = _Dataset(paths)
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    np.savez_compressed(
        sidecars / "0000000.0000000.npz",
        stability_mask=np.asarray([True, True]),
    )
    sampler = ParentPairBatchSampler(
        dataset,
        batch_size=4,
        shuffle=False,
        pair_source_markers=["QM9PBEForceRandom1000PairedTrain"],
        infer_pair_metadata_from_filename=True,
        pair_replay_sidecar_dir=sidecars,
        pair_replay_interval=4,
        pair_replay_phase=0,
        hvp_replay_sidecar_dir=sidecars,
        hvp_replay_per_batch=True,
        hvp_replay_interval=4,
        hvp_replay_phase=1,
    )
    batches = list(sampler)
    pair = set(sampler.pairs[0])
    hvp_index = sampler.hvp_replay_indices[0]
    pair_batches = [batch for batch in batches if pair.issubset(batch)]
    hvp_batches = [batch for batch in batches if hvp_index in batch]

    assert pair_batches and hvp_batches
    assert all(hvp_index not in batch for batch in pair_batches)
    assert all(not pair.intersection(batch) for batch in hvp_batches)
    assert all(len(batch) == 4 for batch in batches)
