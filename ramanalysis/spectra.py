from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from natsort import natsorted
from numpy.typing import NDArray
from scipy.signal import find_peaks, medfilt

from .calibrate import _OpenRamanDataCalibrator, _OpenRamanDataProcessor
from .peak_fitting import find_n_most_prominent_peaks
from .readers import (
    read_horiba_txt,
    read_openraman_csv,
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
        kernel_size: int = 5,
        rough_calibration_residuals_threshold: float = 1.0,
        fine_calibration_residuals_threshold: float = 1e6,
    ) -> RamanSpectrum:
        """Load the calibrated Raman spectrum from a CSV file output by the OpenRAMAN spectrometer.

        Args:
            csv_filepath:
                File path to the input OpenRAMAN CSV file.
            csv_filepath_excitation_calibration:
                File path to the excitation calibration CSV file.
            csv_filepath_emission_calibration:
                File path to the emission calibration CSV file.
            excitation_wavelength_nm:
                Wavelength (in nanometers) of the excitation light source. Default is 532 nm as
                that is the wavelength of the diode laser that the OpenRAMAN system is currently
                equipped with.
            kernel_size:
                Kernel size of the median filter to be applied to the calibration spectra. Must be
                an odd integer. Set `kernel_size` to 1 to avoid smoothing. Default size is 5. Note
                that no smoothing is applied to the input OpenRAMAN data.
        """
        wavenumbers_cm1, intensities = _OpenRamanDataProcessor(
            csv_filepath,
            csv_filepath_excitation_calibration,
            csv_filepath_emission_calibration,
            excitation_wavelength_nm,
            kernel_size,
            rough_calibration_residuals_threshold,
            fine_calibration_residuals_threshold,
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

    def smooth(self, kernel_size: int = 5) -> RamanSpectrum:
        """"""
        smoothed_intensities = medfilt(self.intensities, kernel_size=kernel_size)
        return RamanSpectrum(self.wavenumbers_cm1, smoothed_intensities)

    def find_n_most_prominent_wavenumbers(
        self,
        num_peaks: int,
        prominence_increment: float = 0.005,
        max_iterations: int = 500,
    ) -> FloatArray:
        """"""
        peak_indices = find_n_most_prominent_peaks(
            self.normalize().intensities, num_peaks, prominence_increment, max_iterations
        )
        return self.wavenumbers_cm1[peak_indices]

    def find_prominent_wavenumbers(self, prominence: float = 0.01, **kwargs) -> FloatArray:
        """"""
        peak_indices = find_peaks(self.intensities, prominence=prominence, **kwargs)
        return self.wavenumbers_cm1[peak_indices]  # type: ignore


@dataclass
class RamanSpectra:
    """"""

    spectra: dict[str, RamanSpectrum]

    @classmethod
    def from_openraman_directory_dirty(
        cls,
        filepath: Path | str,
        sample_glob_str: str = "*.csv",
        excitation_glob_str: str = "*neon*.csv",
        emission_glob_str: str = "*aceto*.csv",
        excitation_wavelength_nm: float = 532,
        kernel_size: int = 5,
        rough_calibration_residuals_threshold: float = 1.0,
        fine_calibration_residuals_threshold: float = 1e6,
    ) -> RamanSpectra:
        """"""

        csv_filepaths_samples = natsorted(Path(filepath).glob(sample_glob_str))
        csv_filepath_excitation_calibration = next(Path(filepath).glob(excitation_glob_str))
        csv_filepath_emission_calibration = next(Path(filepath).glob(emission_glob_str))

        wavenumbers_cm1 = _OpenRamanDataCalibrator(
            csv_filepath_excitation_calibration,
            csv_filepath_emission_calibration,
            excitation_wavelength_nm,
            kernel_size,
            rough_calibration_residuals_threshold,
            fine_calibration_residuals_threshold,
        ).calibrate()

        spectra = {}
        for csv_filepath in csv_filepaths_samples:
            sample_name = csv_filepath.stem
            intensities = read_openraman_csv(csv_filepath)
            spectra[sample_name] = RamanSpectrum(wavenumbers_cm1, intensities)

        return RamanSpectra(spectra)
