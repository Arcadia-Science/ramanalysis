import numpy as np
import pytest

from ramanalysis.calibrate import (
    ACETONITRILE_PEAKS_CM1,
    EXCITATION_WAVELENGTH_NM,
    FINE_CALIBRATION_RESIDUALS_THRESHOLD,
    KERNEL_SIZE,
    ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
    _OpenRamanDataCalibrator,
    calculate_raman_shift,
)
from ramanalysis.peak_fitting import find_n_most_prominent_peaks

# set tolerance for testing
FINE_CALIBRATION_TOLERANCE_CM1 = 4


def test_calculate_raman_shift_basic():
    """Test :func:`calculate_raman_shift` with a non-default excitation wavelength.

    See also:
        - https://www.photonetc.com/raman-shift-calculator
    """
    emission_wavelengths = np.array([800, 850, 900], dtype=float)
    excitation_wavelength = 785
    calculated_shifts = calculate_raman_shift(emission_wavelengths, excitation_wavelength)
    # calculated via the definition for Raman shift: (1 / excitation - 1 / emission )
    # and converted to cm^-1
    expected_shifts = [238.85, 974.15, 1627.74]
    np.testing.assert_allclose(calculated_shifts, expected_shifts, rtol=1e-4)


def test_calculate_raman_shift_invalid_input():
    """Test :func:`calculate_raman_shift` with invalid input types."""
    with pytest.raises(TypeError):
        calculate_raman_shift("not an array", 532)  # type: ignore
    with pytest.raises(TypeError):
        calculate_raman_shift(np.array([600, 650]), "invalid wavelength")  # type: ignore


def test_rough_calibration_fails(
    valid_openraman_excitation_csv_filepath,
    valid_openraman_emission_csv_filepath,
):
    """Integration test for :func:`calibrate_rough` on an individual pair of excitation and
    emission spectra.

    Expectation:
        A `ValueError` is raised because the sum of squared residuals from the least-squares fit
        is higher than the specified threshold, which has been decreased for this test by 100x.
    """
    calibrator = _OpenRamanDataCalibrator(
        valid_openraman_excitation_csv_filepath,
        valid_openraman_emission_csv_filepath,
        excitation_wavelength_nm=EXCITATION_WAVELENGTH_NM,
        kernel_size=KERNEL_SIZE,
        rough_calibration_residuals_threshold=ROUGH_CALIBRATION_RESIDUALS_THRESHOLD / 100,
        fine_calibration_residuals_threshold=FINE_CALIBRATION_RESIDUALS_THRESHOLD,
    )
    with pytest.raises(ValueError):
        calibrator.calibrate_rough()


def test_fine_calibration_fails(
    valid_openraman_excitation_csv_filepath,
    valid_openraman_emission_csv_filepath,
):
    """Integration test for :func:`calibrate_fine` on an individual pair of excitation and
    emission spectra.

    Expectation:
        A `ValueError` is raised because the sum of squared residuals from the least-squares fit
        is higher than the specified threshold, which has been decreased for this test by 100x.
    """
    calibrator = _OpenRamanDataCalibrator(
        valid_openraman_excitation_csv_filepath,
        valid_openraman_emission_csv_filepath,
        excitation_wavelength_nm=EXCITATION_WAVELENGTH_NM,
        kernel_size=KERNEL_SIZE,
        rough_calibration_residuals_threshold=ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
        fine_calibration_residuals_threshold=FINE_CALIBRATION_RESIDUALS_THRESHOLD / 100,
    )
    wavenumbers_cm1_rough = calibrator.calibrate_rough()
    with pytest.raises(ValueError):
        calibrator.calibrate_fine(wavenumbers_cm1_rough)


def test_rough_calibration_batch_succeeds(
    batch_neon_spectra_csv_filepaths,
    batch_acetonitrile_spectra_csv_filepaths,
):
    """Test :func:`calibrate_rough` on a batch of excitation spectra.

    Expectation:
        Test passes because the sum of squared residuals from the least-squares fit is lower than
        the specified threshold for every excitation spectrum in the batch. This essentially
        means the fit was "good enough" in all cases.
    """
    for excitation_csv_filepath, emission_csv_filepath in zip(
        batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
    ):
        calibrator = _OpenRamanDataCalibrator(
            excitation_csv_filepath,
            emission_csv_filepath,
            excitation_wavelength_nm=EXCITATION_WAVELENGTH_NM,
            kernel_size=KERNEL_SIZE,
            rough_calibration_residuals_threshold=ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
            fine_calibration_residuals_threshold=FINE_CALIBRATION_RESIDUALS_THRESHOLD,
        )
        calibrator.calibrate_rough()


def test_fine_calibration_batch_succeeds(
    batch_neon_spectra_csv_filepaths,
    batch_acetonitrile_spectra_csv_filepaths,
):
    """Integration test for :func:`calibrate_fine` on a batch of paired excitation and
    emission spectra.

    Expectation:
        Test passes because the calibrated acetonitrile peaks detected from each spectrum in the
        batch are are within the specified tolerance.
    """
    for excitation_csv_filepath, emission_csv_filepath in zip(
        batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
    ):
        calibrator = _OpenRamanDataCalibrator(
            excitation_csv_filepath,
            emission_csv_filepath,
            excitation_wavelength_nm=EXCITATION_WAVELENGTH_NM,
            kernel_size=KERNEL_SIZE,
            rough_calibration_residuals_threshold=ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
            fine_calibration_residuals_threshold=FINE_CALIBRATION_RESIDUALS_THRESHOLD,
        )
        wavenumbers_cm1_rough = calibrator.calibrate_rough()
        wavenumbers_cm1 = calibrator.calibrate_fine(wavenumbers_cm1_rough)

        # verify that calibrated wavenumbers are within the specified tolerance
        emission_intensities = calibrator.emission_intensities
        detected_peak_indices = find_n_most_prominent_peaks(
            emission_intensities, ACETONITRILE_PEAKS_CM1.size
        )
        np.testing.assert_allclose(
            ACETONITRILE_PEAKS_CM1,
            wavenumbers_cm1[detected_peak_indices],
            atol=FINE_CALIBRATION_TOLERANCE_CM1,
        )


def test_fine_calibration_with_refined_peaks_batch_succeeds(
    batch_neon_spectra_csv_filepaths,
    batch_acetonitrile_spectra_csv_filepaths,
):
    """Integration test for :func:`calibrate_fine` on a batch of paired excitation and
    emission spectra.

    Expectation:
        Test passes because the calibrated acetonitrile peaks detected from each spectrum in the
        batch are are within the specified tolerance.
    """
    for excitation_csv_filepath, emission_csv_filepath in zip(
        batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
    ):
        calibrator = _OpenRamanDataCalibrator(
            excitation_csv_filepath,
            emission_csv_filepath,
            excitation_wavelength_nm=EXCITATION_WAVELENGTH_NM,
            kernel_size=KERNEL_SIZE,
            rough_calibration_residuals_threshold=ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
            fine_calibration_residuals_threshold=FINE_CALIBRATION_RESIDUALS_THRESHOLD,
        )
        wavenumbers_cm1_rough = calibrator.calibrate_rough()
        wavenumbers_cm1 = calibrator.calibrate_fine_with_refined_peaks(wavenumbers_cm1_rough)

        # verify that calibrated wavenumbers are within the specified tolerance
        emission_intensities = calibrator.emission_intensities
        detected_peak_indices = find_n_most_prominent_peaks(
            emission_intensities, ACETONITRILE_PEAKS_CM1.size
        )
        np.testing.assert_allclose(
            ACETONITRILE_PEAKS_CM1,
            wavenumbers_cm1[detected_peak_indices],
            atol=FINE_CALIBRATION_TOLERANCE_CM1,
        )
