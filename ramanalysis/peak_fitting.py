import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# type aliases
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
ScalarArray = FloatArray | IntArray


def gaussian(x: ScalarArray, amplitude: float, mean: float, stddev: float) -> FloatArray:
    """1D Gaussian distribution."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev**2))


def refine_peak(peak: int, signal: FloatArray, window_size: int = 11) -> float:
    """Refine an estimated peak position in a 1D signal by least-squares Gaussian fit.

    Args:
        peak: Integer index position of `signal` at which a peak is expected.
        signal: Input one-dimensional signal with which to refine the peak position.
        window_size: Window size around the peak for Gaussian fit.
    """
    peak = int(peak)
    x_window_start, x_window_end = (peak - window_size // 2, peak + window_size // 2 + 1)
    x_window = np.arange(x_window_start, x_window_end)
    initial_guesses = [1, peak, 1]
    fit_parameters, _covariance_matrix = curve_fit(
        gaussian, xdata=x_window, ydata=signal[x_window], p0=initial_guesses
    )
    return fit_parameters[1]


def refine_peaks(peaks: IntArray, signal: FloatArray, window_size: int = 11) -> FloatArray:
    """Applies :func:`refine_peak` to an arbitrary number of peaks."""
    refined_peaks: list[float] = []
    for peak in peaks:
        refined_peak = refine_peak(peak, signal, window_size)
        refined_peaks.append(refined_peak)
    return np.array(refined_peaks)


def find_n_most_prominent_peaks(
    signal: FloatArray,
    num_peaks: int,
    prominence_increment: float = 0.005,
    max_iterations: int = 500,
) -> IntArray:
    """Find specified number of peaks from a 1D signal.

    Optimization loop that starts by finding the max number of peaks (no prominence) and then
    increases the prominence with each iteration to filter out less prominent peaks until only the
    specified number of peaks remains.

    Args:
        signal: Input one-dimensional signal from which to find peaks
        num_peaks: Specified number of peaks to find.
        prominence_increment: Increment for prominence value in the optimization loop.
        max_iterations: Max number of iterations for the optimization loop.

    See also:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        - https://stackoverflow.com/a/52612432/5285918
        - https://en.wikipedia.org/wiki/Topographic_prominence
    """
    i = 0
    prominence = 0
    peaks, _ = find_peaks(signal, prominence=prominence)
    num_peaks_found = int(peaks.size)  # type: ignore
    if num_peaks_found < num_peaks:
        message = (
            f"The number of peaks found with minimal prominence ({num_peaks_found}) is less than "
            f"the specified number of peaks ({num_peaks})."
        )
        logger.warning(message)
    else:
        while (num_peaks_found > num_peaks) and (i < max_iterations):
            peaks, _ = find_peaks(signal, prominence=prominence)
            num_peaks_found = int(peaks.size)  # type: ignore
            prominence += prominence_increment
            i += 1

        if num_peaks_found > num_peaks:
            message = (
                "Max iterations reached before finding the specified number of most prominent "
                "peaks. Try increasing `max_iterations` or `prominence_increment`."
            )
            logger.warning(message)
        elif num_peaks_found < num_peaks:
            message = (
                f"The number of peaks found ({num_peaks_found}) is less than the specified number "
                f"of peaks ({num_peaks}). Try decreasing `prominence_increment` to reduce the "
                "chance that peaks with similar prominences are skipped over."
            )
            logger.warning(message)
        else:  # num_peaks_found == num_peaks --> successful
            pass

    return np.array(peaks)
