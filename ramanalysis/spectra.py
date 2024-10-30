from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from natsort import natsorted
from numpy.typing import NDArray
from scipy.signal import medfilt

from .peak_fitting import find_n_most_prominent_peaks
from .readers import (
    _OpenRamanDataCalibrator,
    _OpenRamanDataProcessor,
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

    def find_n_most_prominent_peaks(self, num_peaks: int) -> FloatArray:
        """"""
        peak_indices = find_n_most_prominent_peaks(self.intensities, num_peaks)
        return self.wavenumbers_cm1[peak_indices]

    # TODO: implement a generic peak finding alorithm based on the ?% most prominent peaks
    # def find_most_prominent_peaks(self, percentile):
    #     peak_indices = find_most_prominent_peaks(self.intensities)
    #     return self.wavenumbers_cm1[peak_indices]


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

        wavenumbers_cm1 = _OpenRamanDataCalibrator(
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
