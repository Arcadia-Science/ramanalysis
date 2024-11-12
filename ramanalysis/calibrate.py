from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import medfilt

from .peak_fitting import (
    find_n_most_prominent_peaks,
    refine_peaks,
)
from .readers import read_openraman_csv
from .utils import (
    interpolate_between_two_values,
    rescale_axis_via_least_squares_fit,
)

logger = logging.getLogger(__name__)

# type aliases
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
ScalarArray = FloatArray | IntArray

NEON_PEAKS_NM: ScalarArray = np.array(
    [
        585.249,
        588.189,
        594.483,
        607.434,
        609.616,
        614.306,
        616.359,
        621.728,
        626.649,
        630.479,
        633.443,
        638.299,
        640.225,
        650.653,
        653.288,
    ]
)
ACETONITRILE_PEAKS_CM1: ScalarArray = np.array([918, 1376, 2249, 2942, 2999])


class _OpenRamanDataProcessor:
    """"""

    def __init__(
        self,
        csv_filepath: Path | str,
        csv_filepath_excitation_calibration: Path | str,
        csv_filepath_emission_calibration: Path | str,
        excitation_wavelength_nm: float,
        kernel_size: int,
        rough_calibration_residuals_threshold: float,
        fine_calibration_residuals_threshold: float,
    ) -> None:
        self.csv_filepath = Path(csv_filepath)
        self.csv_filepath_excitation_calibration = Path(csv_filepath_excitation_calibration)
        self.csv_filepath_emission_calibration = Path(csv_filepath_emission_calibration)
        self.excitation_wavelength_nm = excitation_wavelength_nm
        self.kernel_size = kernel_size
        self.rough_calibration_residuals_threshold = rough_calibration_residuals_threshold
        self.fine_calibration_residuals_threshold = fine_calibration_residuals_threshold

    def process(self) -> tuple[FloatArray, FloatArray]:
        """"""
        intensities = read_openraman_csv(self.csv_filepath)
        wavenumbers_cm1 = self.calibrate()
        return wavenumbers_cm1, intensities

    def calibrate(self) -> FloatArray:
        """"""
        wavenumbers_cm1 = _OpenRamanDataCalibrator(
            self.csv_filepath_excitation_calibration,
            self.csv_filepath_emission_calibration,
            self.excitation_wavelength_nm,
            self.kernel_size,
            self.rough_calibration_residuals_threshold,
            self.fine_calibration_residuals_threshold,
        ).calibrate()
        return wavenumbers_cm1


class _OpenRamanDataCalibrator:
    """Handles the calibration procedure for the OpenRAMAN spectrometer.

    Calibration procedure consists of two steps:
        1) A rough calibration based on a broadband excitation light source (e.g. Neon lamp)
        2) A fine calibration based on Raman-scattered light from a standard sample
           (e.g. acetonitrile)

    High-level batch testing did not reveal that spectra were calibrated with higher accuracy by
    refining peak positions, as measured by the differences in calculated vs expected peak
    wavenumbers in the acetonitrile reference spectrum.
    """

    def __init__(
        self,
        csv_filepath_excitation_calibration: Path | str,
        csv_filepath_emission_calibration: Path | str,
        excitation_wavelength_nm: float,
        kernel_size: int,
        rough_calibration_residuals_threshold: float,
        fine_calibration_residuals_threshold: float,
    ) -> None:
        self.csv_filepath_excitation_calibration = Path(csv_filepath_excitation_calibration)
        self.csv_filepath_emission_calibration = Path(csv_filepath_emission_calibration)
        self.kernel_size = kernel_size
        self.excitation_wavelength_nm = excitation_wavelength_nm
        self.rough_calibration_residuals_threshold = rough_calibration_residuals_threshold
        self.fine_calibration_residuals_threshold = fine_calibration_residuals_threshold

        self.excitation_intensities = self.load_normalized_excitation_intensities()
        self.emission_intensities = self.load_normalized_emission_intensities()

    def load_normalized_excitation_intensities(self) -> FloatArray:
        intensities = read_openraman_csv(self.csv_filepath_excitation_calibration)
        intensities_smoothed = medfilt(intensities, kernel_size=self.kernel_size)
        min_intensity = intensities_smoothed.min()  # type: ignore
        max_intensity = intensities_smoothed.max()  # type: ignore
        intensities_normalized = (intensities_smoothed - min_intensity) / (
            max_intensity - min_intensity
        )
        return intensities_normalized

    def load_normalized_emission_intensities(self) -> FloatArray:
        intensities = read_openraman_csv(self.csv_filepath_emission_calibration)
        intensities_smoothed = medfilt(intensities, kernel_size=self.kernel_size)
        min_intensity = intensities_smoothed.min()  # type: ignore
        max_intensity = intensities_smoothed.max()  # type: ignore
        intensities_normalized = (intensities_smoothed - min_intensity) / (
            max_intensity - min_intensity
        )
        return intensities_normalized

    def calibrate(self) -> FloatArray:
        """Rough calibration + fine calibration."""
        wavenumbers_cm1_rough = self.calibrate_rough()
        wavenumbers_cm1 = self.calibrate_fine(wavenumbers_cm1_rough)
        return wavenumbers_cm1

    def calibrate_rough(self) -> FloatArray:
        """Perform the rough calibration procedure."""
        # convert camera pixel indices to wavelengths (nm) using known peaks in Neon spectrum
        reference_peaks_nm = NEON_PEAKS_NM
        detected_peaks_indices = find_n_most_prominent_peaks(
            self.excitation_intensities, num_peaks=reference_peaks_nm.size
        )
        initial_range = np.arange(self.excitation_intensities.size)
        spectral_range_nm, fitness = rescale_axis_via_least_squares_fit(
            initial_range,
            detected_peaks_indices,
            reference_peaks_nm,
            degree=1,
        )

        # check that sum of squared residuals < specified threshold
        sum_of_squared_residuals = fitness[0].item()  # type: ignore
        if sum_of_squared_residuals > self.rough_calibration_residuals_threshold:
            msg = (
                "Sum of squared residuals during rough calibration > specified threshold "
                f"({sum_of_squared_residuals:.2g} > {self.rough_calibration_residuals_threshold})."
            )
            raise ValueError(msg)

        # calculate Raman shift: nm --> cm^-1
        wavenumbers_cm1_rough = calculate_raman_shift(
            spectral_range_nm, excitation_wavelength_nm=self.excitation_wavelength_nm
        )
        return wavenumbers_cm1_rough

    def calibrate_fine(self, wavenumbers_cm1_rough: FloatArray) -> FloatArray:
        """Perform the fine calibration procedure."""
        # convert rough wavenumbers (cm^-1) to more accurate wavenumbers (cm^-1) using known peaks
        # in the acetonitrile spectrum
        reference_peaks_cm1 = ACETONITRILE_PEAKS_CM1
        detected_peaks_indices = find_n_most_prominent_peaks(
            self.emission_intensities, num_peaks=reference_peaks_cm1.size
        )
        detected_peaks_cm1 = wavenumbers_cm1_rough[detected_peaks_indices]
        wavenumbers_cm1, fitness = rescale_axis_via_least_squares_fit(
            wavenumbers_cm1_rough, detected_peaks_cm1, reference_peaks_cm1
        )

        # check that sum of squared residuals < specified threshold
        sum_of_squared_residuals = fitness[0].item()  # type: ignore
        if sum_of_squared_residuals > self.fine_calibration_residuals_threshold:
            msg = (
                "Sum of squared residuals during fine calibration > specified threshold "
                f"({sum_of_squared_residuals:.2g} > {self.fine_calibration_residuals_threshold})."
            )
            raise ValueError(msg)

        return wavenumbers_cm1

    def calibrate_fine_with_refined_peaks(
        self,
        wavenumbers_cm1_rough: FloatArray,
        method: str = "parabolic",
    ) -> FloatArray:
        """Perform the fine calibration procedure with subpixel interpolation of peak positions."""
        reference_peaks_cm1 = ACETONITRILE_PEAKS_CM1
        detected_peaks_indices = find_n_most_prominent_peaks(
            self.emission_intensities, num_peaks=reference_peaks_cm1.size
        )
        refined_peaks_float_indices = refine_peaks(
            peaks=detected_peaks_indices,
            signal=self.emission_intensities,
            method=method,
        )
        # split float indices into integer and fraction components for interpolation
        refined_peaks_int_indices = np.floor(refined_peaks_float_indices).astype(int)
        refined_peaks_fractional_indices = refined_peaks_float_indices % 1
        refined_peaks_cm1 = interpolate_between_two_values(
            wavenumbers_cm1_rough[refined_peaks_int_indices],
            wavenumbers_cm1_rough[refined_peaks_int_indices + 1],
            refined_peaks_fractional_indices,
        )
        wavenumbers_cm1, fitness = rescale_axis_via_least_squares_fit(
            wavenumbers_cm1_rough, np.array(refined_peaks_cm1), reference_peaks_cm1
        )

        # check that sum of squared residuals < specified threshold
        sum_of_squared_residuals = fitness[0].item()  # type: ignore
        if sum_of_squared_residuals > self.fine_calibration_residuals_threshold:
            msg = (
                "Sum of squared residuals during fine calibration with refined peaks > "
                "specified threshold "
                f"({sum_of_squared_residuals:.2g} > {self.fine_calibration_residuals_threshold})."
            )
            raise ValueError(msg)

        return wavenumbers_cm1


def calculate_raman_shift(
    emission_wavelengths_nm: FloatArray,
    excitation_wavelength_nm: float = 532,
) -> FloatArray:
    """Calculate the Raman shift (cm^-1) for a range of wavelengths (nm).

    The Raman shift corresponds to the energy difference between vibrational or rotational energy
    levels in the molecules of a sample. It is calculated as the difference in wavenumber (cm^-1)
    between an incident light source (usually a laser) and the Raman-scattered light emitted from
    the sample.
    """
    raman_shift = (1 / excitation_wavelength_nm - 1 / emission_wavelengths_nm) * 1e7
    return raman_shift
