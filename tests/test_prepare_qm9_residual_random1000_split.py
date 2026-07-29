from scripts.prepare_qm9_residual_random1000_split import (
    EXPECTED_PARENT_HASHES,
    PARTITION_SIZES,
    _parent_hash,
    historical_parent_split,
)


def test_historical_random1000_parent_split_is_frozen_and_disjoint():
    split = historical_parent_split()

    assert {name: len(ids) for name, ids in split.items()} == PARTITION_SIZES
    assert {_parent_hash(ids) for ids in split.values()} == set(
        EXPECTED_PARENT_HASHES.values()
    )
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    assert set().union(*map(set, split.values())) == set(range(1, 1001))


def test_historical_random1000_split_keeps_known_legacy_prefixes():
    split = historical_parent_split()

    assert split["train"][:10] == [909, 266, 628, 328, 245, 616, 626, 585, 926, 566]
    assert split["val"][:10] == [981, 844, 173, 931, 118, 745, 122, 215, 453, 448]
    assert split["test"][:10] == [119, 207, 113, 529, 305, 964, 20, 427, 743, 394]
