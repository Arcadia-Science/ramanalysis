from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt

from .calibrate import (
    EXCITATION_WAVELENGTH_NM,
    FINE_CALIBRATION_RESIDUALS_THRESHOLD,
    KERNEL_SIZE,
    ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
    _OpenRamanDataProcessor,
)
from .peak_fitting import find_n_most_prominent_peaks
from .readers import (
    read_horiba_txt,
    read_renishaw_singlepoint_txt,
    read_wasatch_csv,
)
from .typing import FloatArray, FloatOrArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RamanSpectrum:
    """A dataclass for Raman spectroscopy data."""

    wavenumbers_cm1: FloatArray
    intensities: FloatArray

    def __post_init__(self):
        # ensure inputs are numpy float arrays
        object.__setattr__(
            self, "wavenumbers_cm1", np.asarray(self.wavenumbers_cm1, dtype=np.float64)
        )
        object.__setattr__(self, "intensities", np.asarray(self.intensities, dtype=np.float64))

        # validate no NaN values
        if np.isnan(self.wavenumbers_cm1).any():
            raise ValueError("Wavenumber data must not contain NaN values.")
        if np.isnan(self.intensities).any():
            raise ValueError("Intensity data must not contain NaN values.")

        # validate array sizes
        if self.wavenumbers_cm1.size == 0:
            raise ValueError("Received empty array for wavenumber data.")
        if self.intensities.size == 0:
            raise ValueError("Received empty array for intensity data.")
        if self.wavenumbers_cm1.ndim != 1:
            raise ValueError("Wavenumber data must be one-dimensional and array-like.")
        if self.intensities.ndim < 1:
            raise ValueError("Intensity data must be at least one-dimensional and array-like.")
        if (self.intensities.ndim == 1) and not (
            self.wavenumbers_cm1.size == self.intensities.size
        ):
            msg = (
                "Wavenumbers and intensities must be the same size when intensities are "
                f"one-dimensional arrays, but received sizes {self.wavenumbers_cm1.size} and "
                f"{self.intensities.size}."
            )
            raise ValueError(msg)

        # validate that wavenumber data is monotonically increasing
        if not np.all(self.wavenumbers_cm1[:-1] <= self.wavenumbers_cm1[1:]):
            raise ValueError("Wavenumber data must be monotonically increasing.")

    @classmethod
    def from_openraman_csvfiles(
        cls,
        csv_filepath: Path | str,
        csv_filepath_excitation_calibration: Path | str,
        csv_filepath_emission_calibration: Path | str,
        excitation_wavelength_nm: float = EXCITATION_WAVELENGTH_NM,
        kernel_size: int = KERNEL_SIZE,
        rough_calibration_residuals_threshold: float = ROUGH_CALIBRATION_RESIDUALS_THRESHOLD,
        fine_calibration_residuals_threshold: float = FINE_CALIBRATION_RESIDUALS_THRESHOLD,
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
    def from_horiba_txtfile(
        cls,
        txt_filepath: Path | str,
        num_skip_rows: int = 32,
    ) -> RamanSpectrum:
        """Load a Raman spectrum from a TXT file output by the Horiba MacroRam or LabRAM."""
        wavenumbers_cm1, intensities, _metadata = read_horiba_txt(txt_filepath, num_skip_rows)
        return RamanSpectrum(wavenumbers_cm1, intensities)

    @classmethod
    def from_renishaw_txtfile(cls, csv_filepath: Path | str) -> RamanSpectrum:
        """Load a Raman spectrum from a TXT file output by the Renishaw Qontor.

        Only supports data from a single point scan.

        See also:
            - To instantiate :obj:`RamanSpectra` from a multipoint scan,
            see :func:`RamanSpectra.from_renishaw_txtfile`.
        """
        wavenumbers_cm1, intensities = read_renishaw_singlepoint_txt(csv_filepath)
        return RamanSpectrum(wavenumbers_cm1, intensities)

    @classmethod
    def from_wasatch_csvfile(cls, csv_filepath: Path | str) -> RamanSpectrum:
        """Load a Raman spectrum from a CSV file output by the Wasatch WP 785X."""
        wavenumbers_cm1, intensities, _metadata = read_wasatch_csv(csv_filepath)
        return RamanSpectrum(wavenumbers_cm1, intensities)

    @classmethod
    def from_generic_csvfile(
        cls,
        csv_filepath: Path | str,
        wavenumber_cm1_index: int = 0,
        intensity_index: int = 1,
        **kwargs,
    ) -> RamanSpectrum:
        """"""
        data = pd.read_csv(csv_filepath, **kwargs).values  # type: ignore
        return RamanSpectrum(wavenumbers_cm1=data[:, 0], intensities=data[:, 1])

    @property
    def snr(self) -> float:
        raise NotImplementedError("TODO: SNR calculation hasn't been implemented.")

    def between(self, min_wavenumber_cm1: float, max_wavenumber_cm1: float) -> RamanSpectrum:
        """Clip spectrum to a specified spectral range."""
        mask = (self.wavenumbers_cm1 > min_wavenumber_cm1) & (
            self.wavenumbers_cm1 < max_wavenumber_cm1
        )
        if ~mask.any():
            msg = (
                f"No spectral data within the specified clipping range ({min_wavenumber_cm1}, "
                f"{max_wavenumber_cm1})."
            )
            raise ValueError(msg)
        else:
            return RamanSpectrum(self.wavenumbers_cm1[mask], self.intensities[mask])

    def interpolate(
        self,
        float_indices: FloatOrArray,
    ) -> tuple[FloatOrArray, FloatOrArray]:
        """One-dimensional linear interpolation.

        Examples:
            Get wavenumbers and intensities corresponding to detected peak positions:

            >>> from ramanalysis.peak_fitting import find_n_most_prominent_peaks
            >>> t = np.linspace(0, np.pi * 5, 1000)
            >>> signal = np.sin(np.pi/2 * t) + np.sin(np.pi/3 * t)
            >>> spectrum = RamanSpectrum(wavenumbers_cm1=t + 200, intensities=signal)
            >>> i_peaks = find_n_most_prominent_peaks(spectrum.intensities, num_peaks=4)
            >>> peaks_cm1, peak_heights = spectrum.interpolate(i_peaks)
            >>> peaks_cm1
            array([201.14782915, 205.34605356, 208.58513308, 213.14500229])
            >>> peak_heights
            array([1.90592364, 0.22333604, 1.21598056, 1.90586467])
        """
        interpolated_wavenumbers_cm1 = np.interp(
            float_indices, np.arange(self.wavenumbers_cm1.size), self.wavenumbers_cm1
        )
        interpolated_intensities = np.interp(
            float_indices, np.arange(self.intensities.size), self.intensities
        )
        return (
            cast(FloatOrArray, interpolated_wavenumbers_cm1),
            cast(FloatOrArray, interpolated_intensities),
        )

    def normalize(self) -> RamanSpectrum:
        """Scale intensities with min-max normalization."""
        _min = self.intensities.min()
        _max = self.intensities.max()
        scaled_intensities = (self.intensities - _min) / (_max - _min)
        return RamanSpectrum(self.wavenumbers_cm1, scaled_intensities)

    def standardize(self) -> RamanSpectrum:
        """Scale intensities with mean-std standardization."""
        mean = self.intensities.mean()
        std = self.intensities.std()
        scaled_intensities = (self.intensities - mean) / std
        return RamanSpectrum(self.wavenumbers_cm1, scaled_intensities)

    def smooth(self, kernel_size: int = 5) -> RamanSpectrum:
        """Smooth intensities with median filtering."""
        smoothed_intensities = medfilt(self.intensities, kernel_size=kernel_size)
        return RamanSpectrum(self.wavenumbers_cm1, smoothed_intensities)

    def find_n_most_prominent_wavenumbers(
        self,
        num_peaks: int,
        prominence_increment: float = 0.005,
        max_iterations: int = 500,
    ) -> FloatArray:
        """Find the specified number of peaks and return the corresponding wavenumbers."""
        peak_indices = find_n_most_prominent_peaks(
            self.intensities,
            num_peaks,
            prominence_increment,
            max_iterations,
        )
        return self.wavenumbers_cm1[peak_indices]

    def find_prominent_wavenumbers(self, prominence: float = 0.01, **kwargs) -> FloatArray:
        """Find prominent peaks and return the corresponding wavenumbers.

        Args:
            prominence:
                Threshold for peak prominence, measured as the vertical distance between the peak
                and its lowest surrounding contour line.
            **kwargs:
                Keyword arguments accepted by `scipy.signal.find_peaks`.

        See also:
            - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        """
        peak_indices = find_peaks(
            self.intensities,
            prominence=prominence,
            **kwargs,
        )
        return self.wavenumbers_cm1[peak_indices]  # type: ignore
