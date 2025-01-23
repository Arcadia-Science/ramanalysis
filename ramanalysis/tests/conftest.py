"""
More info regarding these fixtures is provided in
`ramanalysis/tests/example_data/README.md`
"""

from pathlib import Path

import numpy as np
import pytest
from natsort import natsorted


@pytest.fixture
def test_data_directory():
    return Path(__file__).parent / "example_data"


@pytest.fixture
def valid_openraman_csv_filepath(test_data_directory):
    return test_data_directory / "OpenRAMAN" / "2024-10-04_CC-124-TAP_n_n_solid_10000_0_5.csv"


@pytest.fixture
def valid_openraman_excitation_csv_filepath(test_data_directory):
    return test_data_directory / "OpenRAMAN" / "2024-10-04_neon_n_n_solid_10000_0_5.csv"


@pytest.fixture
def valid_openraman_emission_csv_filepath(test_data_directory):
    return test_data_directory / "OpenRAMAN" / "2024-10-04_acetonitrile_n_n_solid_10000_0_5.csv"


@pytest.fixture
def valid_horiba_singlepoint_txt_filepath(test_data_directory):
    return test_data_directory / "Horiba" / "MacroRAM" / "polystyrene.txt"


@pytest.fixture
def valid_renishaw_singlepoint_filepath(test_data_directory):
    return test_data_directory / "Renishaw" / "CC-125_TAP_singlepoint.txt"


@pytest.fixture
def valid_renishaw_multipoint_filepath(test_data_directory):
    return test_data_directory / "Renishaw" / "CC-125_TAP_multipoint.txt"


@pytest.fixture
def invalid_renishaw_singlepoint_filepath(test_data_directory):
    return test_data_directory / "Renishaw" / "CC-125_TAP_singlepoint_corrupt.txt"


@pytest.fixture
def invalid_renishaw_multipoint_filepath(test_data_directory):
    return test_data_directory / "Renishaw" / "CC-125_TAP_multipoint_corrupt.txt"


@pytest.fixture
def valid_wasatch_csv_filepath(test_data_directory):
    return test_data_directory / "Wasatch" / "acetonitrile.csv"


@pytest.fixture
def valid_synthetic_spectrum_wavenumbers(test_data_directory):
    valid_synthetic_spectrum_filepath = (
        test_data_directory / "synthetic_data" / "fake_spectrum_0.txt"
    )
    synthetic_wavenumbers_cm1 = np.loadtxt(valid_synthetic_spectrum_filepath)[:, 0]
    return synthetic_wavenumbers_cm1


@pytest.fixture
def valid_synthetic_spectrum_intensities(test_data_directory):
    valid_synthetic_spectrum_filepath = (
        test_data_directory / "synthetic_data" / "fake_spectrum_0.txt"
    )
    synthetic_intensities = np.loadtxt(valid_synthetic_spectrum_filepath)[:, 1]
    return synthetic_intensities


@pytest.fixture
def batch_neon_spectra_csv_filepaths(test_data_directory: Path):
    """File paths to a batch of neon spectra independently acquired over several weeks."""
    calibration_data_filepath = test_data_directory / "OpenRAMAN" / "calibration_data"
    batch_neon_spectra_csv_filepaths = natsorted(calibration_data_filepath.glob("*neon*.csv"))
    return batch_neon_spectra_csv_filepaths


@pytest.fixture
def batch_acetonitrile_spectra_csv_filepaths(test_data_directory: Path):
    """File paths to a batch of acetonitrile spectra independently acquired over several weeks."""
    calibration_data_filepath = test_data_directory / "OpenRAMAN" / "calibration_data"
    batch_acetonitrile_spectra_csv_filepaths = natsorted(
        calibration_data_filepath.glob("*aceto*.csv")
    )
    return batch_acetonitrile_spectra_csv_filepaths
