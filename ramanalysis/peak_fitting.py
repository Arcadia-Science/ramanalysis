import logging

import numpy as np
from numpy.polynomial import Polynomial
from scipy.signal import find_peaks

from .typing import FloatArray, IntArray, ScalarArray

logger = logging.getLogger(__name__)


def refine_peak_parabolic_fit(
    i_peak: int,
    y_values: FloatArray,
    x_values: ScalarArray | None = None,
) -> tuple[float, float]:
    """Refine an estimated peak position in a 1D signal by a parabolic fit.

    Uses the formula

        i_peak = i + (y_{i-1} - y_{i+1}) / (2 * (y_{i-1} - 2y_i + y_{i+1}))

    where y_{i-1}, y_i, and y_{i+1} are the values of the signal at indices i-1, i, and i+1.
    Assuming the peak is roughly centered around i, the parabola is fit around these three points.

    Args:
        i_peak: Integer index position of `y_values` at which a peak is expected.
        y_values: Input one-dimensional signal with which to refine the peak position.
        x_values: (Optional) input x values corresponding to the input one-dimensional signal.

    Returns:
        x_interpolated_peak: Interpolated peak position in x.
        y_interpolated_peak: Interpolated peak position in y.
    """
    if x_values is None:
        x_values = np.arange(y_values.size)

    # small window around the peak (3 points: left, peak, right)
    if 0 < i_peak < y_values.size - 1:
        i_window = np.array([i_peak - 1, i_peak, i_peak + 1])
        x_window = x_values[i_window]
        y_window = y_values[i_window]
    else:
        msg = "Peak index is at the edge of the signal and therefore cannot be interpolated."
        logger.warning(msg)
        return x_values[i_peak], y_values[i_peak]

    # fit parabola to 3 points
    parabola = Polynomial.fit(  # type: ignore
        x_window, y_window, deg=2
    ).convert()
    a, b, _c = parabola.coef[::-1]
    x_interpolated_peak = -b / (2 * a)

    # sometimes it does a really bad job \_0_/
    if not x_window[0] < x_interpolated_peak < x_window[-1]:
        msg = "Parabolic interpolation failed: Interpolated x position is out of bounds."
        logger.warning(msg)
        return x_values[i_peak], y_values[i_peak]

    y_interpolated_peak = parabola(x_interpolated_peak)
    return x_interpolated_peak, y_interpolated_peak


def refine_peaks(
    i_peaks: IntArray,
    y_values: FloatArray,
    x_values: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Applies :func:`refine_peak_parabolic_fit` to an arbitrary number of peaks.

    Args:
        i_peaks: Integer index positions of `y_values` at which peaks are expected.
        y_values: Input one-dimensional signal with which to refine the peak positions.
        x_values: (Optional) input x values corresponding to the input one-dimensional signal.

    Returns:
        x_refined_peaks: Array of refined peak positions in x.
        y_refined_peaks: Array of refined peak positions in y.
    """
    x_refined_peaks: list[float] = []
    y_refined_peaks: list[float] = []
    for i_peak in i_peaks:
        x_refined_peak, y_refined_peak = refine_peak_parabolic_fit(i_peak, y_values, x_values)
        x_refined_peaks.append(x_refined_peak)
        y_refined_peaks.append(y_refined_peak)
    return np.array(x_refined_peaks), np.array(y_refined_peaks)


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
        signal: Input one-dimensional signal from which to find peaks.
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
    peaks, _properties = find_peaks(signal, prominence=prominence)
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
