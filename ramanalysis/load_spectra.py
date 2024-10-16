from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from numpy.typing import NDArray
from scipy.signal import medfilt

from .calibration import (
    ACETONITRILE_PEAKS_CM1,
    NEON_PEAKS_NM,
    calculate_raman_shift,
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


@dataclass(frozen=True)
class RamanSpectrum:
    """"""

    wavenumbers_cm1: FloatArray
    intensities: FloatArray

    @classmethod
    def from_openraman_csvfiles(
        cls,
        csv_filepath: Path | str,
        csv_filepath_excitation_calibration: Path | str,
        csv_filepath_emission_calibration: Path | str,
        excitation_wavelength_nm: float = 532,
        kernel_size: int = 3,
    ) -> RamanSpectrum:
        """Load a Raman spectrum from a CSV file output by the OpenRAMAN spectrometer.

        Args:
            kernel_size:
                Kernel size of the median filter to be applied to the calibration spectra. Must be
                an odd integer. Set `kernel_size` to 1 to avoid smoothing. Default size is 3.
        """
        wavenumbers_cm1, intensities = _OpenRamanDataProcessor(
            csv_filepath,
            csv_filepath_excitation_calibration,
            csv_filepath_emission_calibration,
            excitation_wavelength_nm,
            kernel_size,
        ).process()

        return RamanSpectrum(wavenumbers_cm1, intensities)

    @classmethod
    def from_horiba_txtfile(cls, txt_filepath: Path | str) -> RamanSpectrum:
        """Load a Raman spectrum from a TXT file output by the Horiba MacroRam."""
        wavenumbers_cm1, intensities = read_horiba_txt(txt_filepath)
        return RamanSpectrum(wavenumbers_cm1, intensities)

    @property
    def snr(self) -> float:
        # TODO: add SNR calculation
        return -1

    def between(self, min_wavenumber_cm1: float, max_wavenumber_cm1: float) -> RamanSpectrum:
        """"""
        wavenumbers_cm1 = self.wavenumbers_cm1[
            (self.wavenumbers_cm1 > min_wavenumber_cm1)
            & (self.wavenumbers_cm1 < max_wavenumber_cm1)
        ]
        intensities = self.intensities[
            (self.wavenumbers_cm1 > min_wavenumber_cm1)
            & (self.wavenumbers_cm1 < max_wavenumber_cm1)
        ]
        return RamanSpectrum(wavenumbers_cm1, intensities)

    def normalize(self) -> RamanSpectrum:
        """"""
        normalized_intensities = (self.intensities - self.intensities.min()) / (
            self.intensities.max() - self.intensities.min()
        )
        return RamanSpectrum(self.wavenumbers_cm1, normalized_intensities)

    def smooth(self, kernel_size: int = 3) -> RamanSpectrum:
        """"""
        smoothed_intensities = medfilt(self.intensities, kernel_size=kernel_size)
        return RamanSpectrum(self.wavenumbers_cm1, smoothed_intensities)


@dataclass
class RamanSpectra:
    """"""

    spectra: dict[str, RamanSpectrum]

    # TODO: implement a cleaner version to instantiate `RamanSpectra` from a directory
    # @classmethod
    # def from_input_directory(cls, filepath: Path | str) -> RamanSpectra:
    #     """"""
    #     pass

    @classmethod
    def from_input_directory_dirty(
        cls,
        filepath: Path | str,
        sample_glob_str: str = "*.csv",
        excitation_glob_str: str = "*neon*.csv",
        emission_glob_str: str = "*aceto*.csv",
        excitation_wavelength_nm: float = 532,
        kernel_size: int = 3,
    ) -> RamanSpectra:
        """"""

        csv_filepaths_samples = natsorted(Path(filepath).glob(sample_glob_str))
        csv_filepath_excitation_calibration = next(Path(filepath).glob(excitation_glob_str))
        csv_filepath_emission_calibration = next(Path(filepath).glob(emission_glob_str))

        wavenumbers_cm1 = _RamanDataCalibrator(
            csv_filepath_excitation_calibration,
            csv_filepath_emission_calibration,
            excitation_wavelength_nm,
            kernel_size,
        ).calibrate()

        spectra = {}
        for csv_filepath in csv_filepaths_samples:
            sample_name = csv_filepath.stem
            intensities = read_openraman_csv(csv_filepath)
            spectra[sample_name] = RamanSpectrum(wavenumbers_cm1, intensities)

        return RamanSpectra(spectra)


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
        wavenumbers_cm1 = _RamanDataCalibrator(
            self.csv_filepath_excitation_calibration,
            self.csv_filepath_emission_calibration,
            self.excitation_wavelength_nm,
            self.kernel_size,
        ).calibrate()
        return wavenumbers_cm1


class _RamanDataCalibrator:
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

    def calibrate_rough(self) -> NDArray[np.float64]:
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
        """Perform the fine calibration procedure (without refining peak positions)."""
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
        """Perform the fine calibration procedure including refining peak positions."""
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
