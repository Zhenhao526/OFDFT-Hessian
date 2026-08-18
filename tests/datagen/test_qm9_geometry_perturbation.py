import numpy as np

from mldft.datagen.datasets.qm9 import QM9GeometryPerturbed
from mldft.datagen.kohn_sham_dataset import load_molecule_sample


def _write_qm9_xyz(path):
    path.write_text(
        "\n".join(
            [
                "2",
                "test molecule",
                "H 0.000000 0.000000 0.000000",
                "H 0.000000 0.000000 0.740000",
                "",
            ]
        )
    )


def _dataset(
    tmp_path,
    num_perturbations=2,
    paired_perturbations=False,
    perturbation_distribution="gaussian",
    remove_translation=True,
):
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir()
    _write_qm9_xyz(raw_data_dir / "dsgdb9nsd_000001.xyz")
    return QM9GeometryPerturbed(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=(tmp_path / "ks").as_posix(),
        label_dir=(tmp_path / "labels").as_posix(),
        filename="qm9_geom_test",
        num_perturbations=num_perturbations,
        perturbation_std=0.01,
        perturbation_distribution=perturbation_distribution,
        max_displacement=0.05,
        random_seed=7,
        remove_translation=remove_translation,
        paired_perturbations=paired_perturbations,
        name="QM9GeometryPerturbedTest",
    )


def test_qm9_geometry_perturbation_samples_are_deterministic(tmp_path):
    dataset = _dataset(tmp_path)

    assert dataset.get_sample_ids(1) == (0, 1, 2)

    charges_0, positions_0, _, _ = dataset.load_sample(1, 0)
    charges_1, positions_1, _, _ = dataset.load_sample(1, 1)
    _, positions_1_again, _, _ = dataset.load_sample(1, 1)

    np.testing.assert_array_equal(charges_0, np.array([1, 1]))
    np.testing.assert_allclose(positions_0, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
    assert not np.allclose(positions_0, positions_1)
    np.testing.assert_allclose(positions_1, positions_1_again)
    np.testing.assert_allclose((positions_1 - positions_0).mean(axis=0), np.zeros(3), atol=1e-12)

    mol = load_molecule_sample(1, dataset, basis="sto-3g", sample_id=1)
    np.testing.assert_allclose(mol.atom_coords(unit="Angstrom"), positions_1)


def test_qm9_geometry_perturbation_done_checks_require_all_samples(tmp_path):
    dataset = _dataset(tmp_path, num_perturbations=1)

    assert dataset.get_chk_file_from_id(1, 0).name == "qm9_geom_test_0000001.0000000.chk"
    assert dataset.get_chk_file_from_id(1, 1).name == "qm9_geom_test_0000001.0000001.chk"
    assert dataset.get_label_file_from_id(1, 0).name == "0000001.0000000.zarr.zip"
    assert dataset.get_label_file_from_id(1, 1).name == "0000001.0000001.zarr.zip"

    dataset.get_chk_file_from_id(1, 0).touch()
    assert dataset.get_ids_done_ks().tolist() == []
    dataset.get_chk_file_from_id(1, 1).touch()
    assert dataset.get_ids_done_ks().tolist() == [1]

    dataset.get_label_file_from_id(1, 0).touch()
    assert dataset.get_ids_done_labelgen().tolist() == []
    dataset.get_label_file_from_id(1, 1).touch()
    assert dataset.get_ids_done_labelgen().tolist() == [1]


def test_qm9_paired_geometry_perturbations_are_exact_opposites(tmp_path):
    dataset = _dataset(tmp_path, num_perturbations=4, paired_perturbations=True)

    _, reference, _, _ = dataset.load_sample(1, 0)
    _, plus_1, _, _ = dataset.load_sample(1, 1)
    _, minus_1, _, _ = dataset.load_sample(1, 2)
    _, plus_2, _, _ = dataset.load_sample(1, 3)
    _, minus_2, _, _ = dataset.load_sample(1, 4)

    np.testing.assert_allclose(plus_1 - reference, -(minus_1 - reference), atol=1e-14)
    np.testing.assert_allclose(plus_2 - reference, -(minus_2 - reference), atol=1e-14)
    assert not np.allclose(plus_1 - reference, plus_2 - reference)


def test_qm9_paired_geometry_perturbations_require_even_count(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "even num_perturbations"):
        _dataset(tmp_path, num_perturbations=3, paired_perturbations=True)


def test_qm9_rademacher_perturbations_are_coordinatewise_sign_vectors(tmp_path):
    dataset = _dataset(
        tmp_path,
        num_perturbations=4,
        paired_perturbations=True,
        perturbation_distribution="rademacher",
        remove_translation=False,
    )

    _, reference, _, _ = dataset.load_sample(1, 0)
    _, plus_1, _, _ = dataset.load_sample(1, 1)
    _, minus_1, _, _ = dataset.load_sample(1, 2)
    _, plus_2, _, _ = dataset.load_sample(1, 3)

    displacement_1 = plus_1 - reference
    displacement_2 = plus_2 - reference
    np.testing.assert_allclose(np.abs(displacement_1), 0.01, atol=1e-14)
    np.testing.assert_allclose(np.abs(displacement_2), 0.01, atol=1e-14)
    np.testing.assert_allclose(displacement_1, -(minus_1 - reference), atol=1e-14)
    assert not np.array_equal(displacement_1, displacement_2)
