import numpy as np

from ramanalysis.spectra import RamanSpectrum


def test_raman_spectrum_from_openraman_csvfiles(
    valid_openraman_csv_filepath,
    valid_openraman_excitation_csv_filepath,
    valid_openraman_emission_csv_filepath,
):
    spectrum = RamanSpectrum.from_openraman_csvfiles(
        valid_openraman_csv_filepath,
        valid_openraman_excitation_csv_filepath,
        valid_openraman_emission_csv_filepath,
    )
    first_five_wavenumbers_cm1 = [402.923577, 404.783446, 406.642939, 408.502055, 410.360796]
    first_five_intensities = [0.661781, 0.661274, 0.664067, 0.658976, 0.660716]
    np.testing.assert_allclose(spectrum.wavenumbers_cm1[:5], first_five_wavenumbers_cm1, rtol=1e-4)
    np.testing.assert_allclose(spectrum.intensities[:5], first_five_intensities, rtol=1e-4)


def test_raman_spectrum_from_horiba_txtfile(valid_horiba_txt_filepath):
    spectrum = RamanSpectrum.from_horiba_txtfile(valid_horiba_txt_filepath)
    first_five_wavenumbers_cm1 = [87.8957, 90.0299, 92.1642, 94.2969, 96.4298]
    first_five_intensities = [620.5, 680.5, 704.5, 705.0, 741.0]
    np.testing.assert_allclose(spectrum.wavenumbers_cm1[:5], first_five_wavenumbers_cm1, rtol=1e-4)
    np.testing.assert_allclose(spectrum.intensities[:5], first_five_intensities, rtol=1e-4)


def test_raman_spectrum_from_synthetic_data(
    valid_synthetic_spectrum_wavenumbers, valid_synthetic_spectrum_intensities
):
    spectrum = RamanSpectrum(
        valid_synthetic_spectrum_wavenumbers, valid_synthetic_spectrum_intensities
    )
    assert spectrum.wavenumbers_cm1.size == 512
    assert spectrum.intensities.size == 512
    assert spectrum.wavenumbers_cm1[0] == 800
