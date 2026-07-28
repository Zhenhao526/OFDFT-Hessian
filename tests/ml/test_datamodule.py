from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.utils.data.distributed import DistributedSampler

from mldft.ml.data.datamodule import OFDataModule


@pytest.mark.parametrize("batch_size", [(7), (32)])
def test_datamodule(batch_size, dummy_basis_info, dummy_dataset_path, master_transform_to_torch):
    """Rudimentary test for the current simple DataModule implementation."""
    label_path = Path(dummy_dataset_path)
    data_path = label_path.parent.parent
    split_file = label_path.parent / "train_val_test_split.pkl"
    datamodule = OFDataModule(
        split_file,
        data_path,
        transforms=master_transform_to_torch,
        basis_info=dummy_basis_info,
        batch_size=batch_size,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    train_loader = datamodule.train_dataloader()
    validation_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    train_length = len(datamodule.train_set)
    validation_length = len(datamodule.val_set)
    test_length = len(datamodule.test_set)
    total_length = train_length + validation_length + test_length
    # Test that each loader is not none, has the correct batch size, and that all positions have 3 dimensions
    for i, loader in enumerate([train_loader, validation_loader, test_loader]):
        assert loader is not None
        assert loader.batch_size == batch_size
        sample = next(iter(loader))
        # If dataset batch size is larger than whole subset, batch size will be smaller.
        if len(loader) > 1:
            assert len(sample) == batch_size
        else:
            assert len(sample) <= batch_size
        assert all(pos.shape[-1] == 3 for pos in sample.pos)


def test_pair_grouped_datamodule_shards_validation_explicitly(
    dummy_basis_info, dummy_dataset_path, master_transform_to_torch
):
    label_path = Path(dummy_dataset_path)
    datamodule = OFDataModule(
        label_path.parent / "train_val_test_split.pkl",
        label_path.parent.parent,
        transforms=master_transform_to_torch,
        basis_info=dummy_basis_info,
        batch_size=4,
        pair_grouped_train_batches=True,
    )
    datamodule.setup("fit")
    datamodule.trainer = SimpleNamespace(world_size=2, global_rank=1)

    loader = datamodule.val_dataloader()

    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 1


def test_train_only_datamodule_never_constructs_held_out_splits(
    dummy_basis_info, dummy_dataset_path, master_transform_to_torch
):
    label_path = Path(dummy_dataset_path)
    datamodule = OFDataModule(
        label_path.parent / "train_val_test_split.pkl",
        label_path.parent.parent,
        transforms=master_transform_to_torch,
        basis_info=dummy_basis_info,
        batch_size=4,
        train_only=True,
    )

    datamodule.setup("fit")

    assert datamodule.train_set is not None
    assert datamodule.val_set is None
    assert datamodule.test_set is None
    with pytest.raises(RuntimeError, match="Validation access is disabled"):
        datamodule.val_dataloader()
    with pytest.raises(RuntimeError, match="Validation access is disabled"):
        datamodule.setup("validate")
    with pytest.raises(RuntimeError, match="Test access is disabled"):
        datamodule.setup("test")
