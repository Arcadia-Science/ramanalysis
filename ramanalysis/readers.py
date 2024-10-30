from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import medfilt

from .peak_fitting import (
    find_n_most_prominent_peaks,
    refine_peaks,
)
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
    ) -> None:
        self.csv_filepath = Path(csv_filepath)
        self.csv_filepath_excitation_calibration = Path(csv_filepath_excitation_calibration)
        self.csv_filepath_emission_calibration = Path(csv_filepath_emission_calibration)
        self.excitation_wavelength_nm = excitation_wavelength_nm
        self.kernel_size = kernel_size

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
        ).calibrate()
        return wavenumbers_cm1


class _OpenRamanDataCalibrator:
    """

    Calibration procedure consists of two steps:
        1) A rough calibration based on a broadband excitation light source (e.g. Neon lamp)
        2) A fine calibration based on Raman-scattered light from a standard sample
           (e.g. acetonitrile)
    """

    def __init__(
        self,
        csv_filepath_excitation_calibration: Path | str,
        csv_filepath_emission_calibration: Path | str,
        excitation_wavelength_nm: float,
        kernel_size: int,
    ) -> None:
        self.csv_filepath_excitation_calibration = Path(csv_filepath_excitation_calibration)
        self.csv_filepath_emission_calibration = Path(csv_filepath_emission_calibration)
        self.kernel_size = kernel_size
        self.excitation_wavelength_nm = excitation_wavelength_nm

        self.excitation_intensities = self.load_excitation_intensities()
        self.emission_intensities = self.load_emission_intensities()

    def load_excitation_intensities(self) -> FloatArray:
        excitation_intensities = medfilt(
            read_openraman_csv(self.csv_filepath_excitation_calibration),
            kernel_size=self.kernel_size,
        )
        return excitation_intensities

    def load_emission_intensities(self) -> FloatArray:
        emission_intensities = medfilt(
            read_openraman_csv(self.csv_filepath_emission_calibration),
            kernel_size=self.kernel_size,
        )
        return emission_intensities

    def calibrate(self) -> FloatArray:
        """Rough calibration + fine calibration with refinement."""
        wavenumbers_cm1_rough = self.calibrate_rough()
        wavenumbers_cm1 = self.calibrate_fine(wavenumbers_cm1_rough)
        # wavenumbers_cm1 = self.calibrate_fine_with_refined_peaks(wavenumbers_cm1_rough)
        return wavenumbers_cm1

    def calibrate_rough(self) -> FloatArray:
        """Perform the rough calibration procedure."""
        reference_peaks_nm = NEON_PEAKS_NM
        detected_peaks_idx = find_n_most_prominent_peaks(
            self.excitation_intensities, num_peaks=reference_peaks_nm.size
        )
        initial_range_idx = np.arange(self.excitation_intensities.size)
        spectral_range_nm, _fitness = rescale_axis_via_least_squares_fit(
            initial_range_idx, detected_peaks_idx, reference_peaks_nm
        )
        wavenumbers_cm1_rough = calculate_raman_shift(
            spectral_range_nm, excitation_wavelength_nm=self.excitation_wavelength_nm
        )
        return wavenumbers_cm1_rough

    def calibrate_fine(self, wavenumbers_cm1_rough: FloatArray) -> FloatArray:
        """Perform the fine calibration procedure without refining peak positions."""
        reference_peaks_cm1 = ACETONITRILE_PEAKS_CM1
        detected_peaks_idx = find_n_most_prominent_peaks(
            self.emission_intensities, num_peaks=reference_peaks_cm1.size
        )
        detected_peaks_cm1 = wavenumbers_cm1_rough[detected_peaks_idx]
        wavenumbers_cm1, _fitness = rescale_axis_via_least_squares_fit(
            wavenumbers_cm1_rough, detected_peaks_cm1, reference_peaks_cm1
        )
        return wavenumbers_cm1

    def calibrate_fine_with_refined_peaks(self, wavenumbers_cm1_rough: FloatArray) -> FloatArray:
        """Perform the fine calibration procedure with subpixel interpolation of peak positions."""
        reference_peaks_cm1 = ACETONITRILE_PEAKS_CM1
        detected_peaks_int_indices = find_n_most_prominent_peaks(
            self.emission_intensities, num_peaks=reference_peaks_cm1.size
        )
        refined_peaks_float_indices = refine_peaks(
            detected_peaks_int_indices, self.emission_intensities
        )
        # split float indices into integer and fraction components for interpolation
        refined_peaks_int_indices = np.floor(refined_peaks_float_indices).astype(int)
        refined_peaks_fractional_indices = refined_peaks_float_indices % 1
        refined_peaks_cm1 = interpolate_between_two_values(
            wavenumbers_cm1_rough[refined_peaks_int_indices],
            wavenumbers_cm1_rough[refined_peaks_int_indices + 1],
            refined_peaks_fractional_indices,
        )
        wavenumbers_cm1, _fitness = rescale_axis_via_least_squares_fit(
            wavenumbers_cm1_rough, np.array([refined_peaks_cm1]), reference_peaks_cm1
        )
        return wavenumbers_cm1


def read_openraman_csv(csv_filepath: Path | str) -> FloatArray:
    """Read data from a CSV file output by the OpenRAMAN.

    Spectral range is not automatically calibrated by the instrument, and thus must be calibrated
    in a subsequent step using known reference values from a standard sample.
    """
    dataframe = pd.read_csv(csv_filepath)  # type: ignore
    return np.array(dataframe["Intensity (a.u.)"].values)


def read_horiba_txt(txt_filepath: Path | str) -> tuple[FloatArray, FloatArray]:
    """Read data from a text file output by the Horiba MacroRam.

    Spectral range is automatically calibrated by the instrument.
    """
    column_names = ["wavenumber_cm-1", "intensity"]
    dataframe = pd.read_csv(  # type: ignore
        txt_filepath,
        skiprows=32,
        header=None,
        names=column_names,
        sep="\t",
        encoding="unicode_escape",
    )
    wavenumbers_cm1 = np.array(dataframe["wavenumber_cm-1"].values)
    intensities = np.array(dataframe["intensity"].values)
    return wavenumbers_cm1, intensities


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
