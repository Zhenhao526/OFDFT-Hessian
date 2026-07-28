from pathlib import Path

import numpy as np

from mldft.utils.create_dataset_splits import (
    get_molecule_id_from_label_path,
    split_grouped,
)


def test_get_molecule_id_from_label_path_supports_sampled_and_plain_labels():
    assert get_molecule_id_from_label_path(Path("0000001.zarr.zip")) == 1
    assert get_molecule_id_from_label_path(Path("0000001.0000000.zarr.zip")) == 1
    assert get_molecule_id_from_label_path(Path("0000123.0000003.zarr")) == 123


def test_split_grouped_keeps_all_samples_of_each_molecule_together():
    group_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    train_indices, val_indices, test_indices = split_grouped(group_ids, (0.5, 0.25, 0.25))

    split_by_group = {}
    for split_name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        for index in indices:
            group_id = group_ids[index]
            if group_id in split_by_group:
                assert split_by_group[group_id] == split_name
            else:
                split_by_group[group_id] = split_name

    assert sorted(split_by_group) == [1, 2, 3, 4]
