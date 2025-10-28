import os
import tempfile
import numpy as np
import pytest
from homogenization.srf_from_population import SRFFromTensorPopulation  # assuming your class is in this module

@pytest.fixture
def fake_config(tmp_path):
    """Create a fake config_dict with .npy tensor and coord files."""
    cond_tns = np.random.rand(10, 3, 3)  # 10 random tensors
    cond_coords = np.random.rand(10, 3) * 10.0  # 10 random coords

    cond_tns_file = tmp_path / "cond_tns.npy"
    cond_coords_file = tmp_path / "cond_coords.npy"

    np.save(cond_tns_file, cond_tns)
    np.save(cond_coords_file, cond_coords)

    config = {
        "fine": {
            "step": 10,
            "cond_tn_pop_file": str(cond_tns_file),
            "cond_tn_pop_coords_file": str(cond_coords_file),
        },
        "coarse": {"step": 30},
        "sim_config": {
            "geometry": {"orig_domain_box": [60.0, 60.0, 60.0]},
            "level_parameters": np.array([[30], [10], [5]]),
        },
    }
    return config


def test_get_larger_domain_size(fake_config):
    larger_size, hom_block_size, previous_level_hom_block_size = SRFFromTensorPopulation.get_larger_domain_size(fake_config)
    assert isinstance(larger_size, (int, float))
    assert larger_size == 115
    assert hom_block_size == 45
    assert previous_level_hom_block_size == 15
    assert larger_size > fake_config["sim_config"]["geometry"]["orig_domain_box"][0]


def test_get_larger_domain_size_the_coarsest_level(fake_config):
    fake_config["coarse"]["step"] = 0
    fake_config["fine"]["step"] = 30
    larger_size, hom_block_size, previous_level_hom_block_size = SRFFromTensorPopulation.get_larger_domain_size(fake_config)
    print("larger_size ", larger_size)
    print("previous_level_hom_block_size ", previous_level_hom_block_size)
    assert hom_block_size == 0
    assert larger_size == fake_config["sim_config"]["geometry"]["orig_domain_box"][0] + previous_level_hom_block_size + 10
    assert isinstance(larger_size, (int, float))
    assert larger_size > fake_config["sim_config"]["geometry"]["orig_domain_box"][0]


def test_expand_domain_centers_1d_valid():
    coords = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    centers = SRFFromTensorPopulation.expand_domain_centers_1d(coords, new_domain=10)
    assert np.allclose(centers, sorted(centers))  # sorted
    assert 0.0 in centers


def test_expand_domain_centers_1d_invalid_shape():
    with pytest.raises(ValueError):
        SRFFromTensorPopulation.expand_domain_centers_1d(np.array([1, 2, 3]), new_domain=10)


def test_calculate_all_centers_shape():
    centers = SRFFromTensorPopulation.calculate_all_centers(
        domain_size=10, block_size=2, overlap=1
    )
    assert centers.shape[1] == 3
    assert centers.ndim == 2


def test_symmetrize_cond_tns():
    cond_tns = np.array([
        [1.0, 2.0, 4.0],
        [5.0, 6.0, 8.0],
    ])
    sym = SRFFromTensorPopulation.symmetrize_cond_tns(cond_tns.copy())
    assert sym.shape[1] == 2
    assert np.allclose(sym[:, 1], [(2 + 4) / 2, (6 + 8) / 2])


def test_constructor_and_centers(fake_config):
    srf = SRFFromTensorPopulation(fake_config)
    assert srf._cond_tns is not None
    assert srf._cond_tns_coords is not None
    assert hasattr(srf, "centers_3d")
    assert srf.centers_3d.shape[1] == 3


def test_generate_field(fake_config):
    srf = SRFFromTensorPopulation(fake_config)
    sampled, centers = srf.generate_field()
    assert sampled.shape[0] == centers.shape[0]
    assert sampled.shape[1:] == (3, 3)

