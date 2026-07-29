from scripts.prepare_qm9_residual_random1000_split import (
    EXPECTED_PARENT_HASHES,
    EXPECTED_RANDOM1000_PARENT_HASH,
    PARTITION_SIZES,
    _parent_hash,
    historical_parent_split,
    historical_random1000_parent_ids,
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
    assert set().union(*map(set, split.values())) == set(
        historical_random1000_parent_ids()
    )
    assert _parent_hash(historical_random1000_parent_ids()) == (
        EXPECTED_RANDOM1000_PARENT_HASH
    )


def test_historical_random1000_split_keeps_known_legacy_prefixes():
    split = historical_parent_split()

    assert historical_random1000_parent_ids()[:10] == [
        23,
        100,
        242,
        253,
        357,
        623,
        751,
        777,
        779,
        835,
    ]
    assert split["train"][:10] == [
        122244,
        37566,
        82549,
        44546,
        35506,
        81113,
        82259,
        76578,
        124009,
        73181,
    ]
    assert split["val"][:10] == [
        131469,
        114642,
        25546,
        125104,
        18347,
        101044,
        18707,
        32059,
        59737,
        59338,
    ]
    assert split["test"][:10] == [
        18436,
        30280,
        17536,
        69992,
        42237,
        129614,
        2686,
        57675,
        100822,
        52985,
    ]
