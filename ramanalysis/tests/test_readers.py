import numpy as np

from ramanalysis.readers import read_horiba_txt, read_openraman_csv


def test_read_openraman_csv(valid_openraman_csv_filepath):
    intensities = read_openraman_csv(valid_openraman_csv_filepath)
    first_five_intensities = [0.661781, 0.661274, 0.664067, 0.658976, 0.660716]
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-6)


def test_read_horiba_txt(valid_horiba_txt_filepath):
    wavenumbers_cm1, intensities = read_horiba_txt(valid_horiba_txt_filepath)
    first_five_wavenumbers = [87.8957, 90.0299, 92.1642, 94.2969, 96.4298]
    first_five_intensities = [620.5, 680.5, 704.5, 705.0, 741.0]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers, rtol=1e-2)
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-2)
