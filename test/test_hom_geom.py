import numpy as np
import pytest
from homogenization.sim_sample_3d import DFMSim3D


@pytest.fixture
def geometry():
    return {
        "orig_domain_box": [60, 60, 60],
        "subdomain_box": [10.0, 10.0, 10.0],
        "pixel_stride_div": 2.0,
    }


def test_calculate_subdomains_fixed(geometry):
    """Test fixed-box subdomain calculation."""
    sub_box, n_nonoverlap, n_axes, centers = DFMSim3D._calculate_subdomains(
        coarse_step=30, geometry=geometry, hom_box_size=None, fixed=True)

    assert isinstance(sub_box, list)
    assert len(sub_box) == 3
    assert n_axes == 4
    assert np.allclose(sub_box, [45, 45, 45])
    assert np.allclose(centers, [0, 20, 40, 60])
    assert n_nonoverlap > 0


def test_calculate_subdomains_adaptive(geometry):
    """Test adaptive-box subdomain calculation."""
    sub_box, n_nonoverlap, n_axes, centers = DFMSim3D._calculate_subdomains(
        coarse_step=30.0, geometry=geometry, hom_box_size=None, fixed=False)

    assert isinstance(sub_box, list)
    assert len(sub_box) == 3
    assert n_nonoverlap == 2
    assert n_axes == 3
    assert np.allclose(sub_box, [60, 60, 60])
    assert centers == []  # adaptive mode returns empty centers


def test_calculate_subdomains_invalid_coarse_step(geometry):
    """Test fallback when coarse_step <= 0."""
    sub_box, n_nonoverlap, n_axes, centers = DFMSim3D._calculate_subdomains(
        coarse_step=0, geometry=geometry, fixed=True
    )
    assert sub_box == geometry["subdomain_box"]
    assert n_nonoverlap == 4
    assert n_axes == 4
    assert centers == []


def test_configure_geometry_with_fine_step_mult(geometry):
    """Test configure_geometry with hom_box_fine_step_mult."""
    config = {"fine": {"step": 10}, "coarse": {"step": 30},
              "sim_config": {"geometry": geometry, "hom_box_fine_step_mult": 3.0}}
    updated_config = DFMSim3D.configure_homogenization_geometry_params(config)

    g = updated_config["sim_config"]["geometry"]
    assert "subdomain_box" in g
    assert "n_subdomains" in g
    assert "hom_block_centers" in g
    assert g["n_subdomains"] == g["n_subdomains_per_axes"] ** 3


def test_configure_geometry_with_fixed_box(geometry):
    """Test configure_geometry with hom_box_fixed=True."""
    config = {"fine": {"step": 10}, "coarse": {"step": 30}, "sim_config": {"geometry": geometry, "hom_box_fixed": True}}
    updated_config = DFMSim3D.configure_homogenization_geometry_params(config)

    g = updated_config["sim_config"]["geometry"]
    assert isinstance(g["hom_block_centers"], np.ndarray)
    assert g["n_subdomains"] == g["n_subdomains_per_axes"] ** 3


def test_configure_geometry_default_path(geometry):
    """Test configure_geometry with default (adaptive) path."""
    config = {"fine": {"step": 10}, "coarse": {"step": 30}, "sim_config": {"geometry": geometry}}
    updated_config = DFMSim3D.configure_homogenization_geometry_params(config)

    g = updated_config["sim_config"]["geometry"]
    assert isinstance(g["hom_block_centers"], list)
    assert g["hom_block_centers"] == []
    assert g["n_subdomains"] == g["n_subdomains_per_axes"] ** 3
