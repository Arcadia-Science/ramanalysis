import logging
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# type aliases
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
ScalarArray = FloatArray | IntArray


def interpolate_between_two_values(
    x0: int | float | ScalarArray,
    x1: int | float | ScalarArray,
    fraction: float | FloatArray,
) -> float | FloatArray:
    """Linear interpolation in one dimension between two points.

    Formula for interpolating between two numbers `x0` and `x1` is
        interpolated value = x0 * fraction + (x1 - x0)
    where `fraction` is a value between 0 and 1 indicating the relative position between
    `x0` and `x1`.

    Examples:
        >>> interpolate(5, 10, 0.5)
        7.5
        >>> x0_arr = np.array([3, 33, 333])
        >>> x1_arr = np.array([4, 44, 444])
        >>> interpolate(x0_arr, x1_arr, 0.1)
        array([  3.1,  34.1, 344.1])
    """
    return x0 + fraction * (x1 - x0)


def rescale_axis_via_least_squares_fit(
    x_range: ScalarArray,
    x_values_observed: ScalarArray,
    x_values_groundtruth: ScalarArray,
    degree: int = 1,
) -> tuple[FloatArray, list]:
    """Rescales an axis by fitting a polynomial transformation to two sets of points.

    This function performs a least-squares polynomial fit between `x_values_observed` and
    `x_values_groundtruth`, which represent corresponding points on the original axis (`x_range`)
    and the desired axis, respectively. The fitted polynomial is then used to transform `x_range`.

    Args:
        x_range:
            The input axis to be rescaled.
        x_values_observed:
            A set of points within the range of the input axis. These are the original values that
            correspond to the ground truth points.
        x_values_groundtruth:
            The ground truth values corresponding to `x_values_observed`. These represent the
            desired values after rescaling.
        degree:
            The degree of the polynomial used for fitting. Default is 1 (linear transformation).

    Examples:
        Perform the first part of the rough calibration procedure which rescales an axis of
        integers corresponding to the number of pixels in one dimension of the camera to
        the spectral range of the Neon light source in nanometers.

        >>> from ramanalysis.load_spectra import read_openraman_csv
        >>> from ramanalysis.calibration import find_n_most_prominent_peaks, NEON_PEAKS_NM
        >>> pixelated_axis = np.arange(2048)
        >>> excitation_intensities = read_openraman_csv("/path/to/excitation_spectrum.csv")
        >>> detected_peaks = find_n_most_prominent_peaks(
            excitation_intensities,
            num_peaks=NEON_PEAKS_NM.size,
        )
        >>> rescale_axis_via_least_squares_fit(pixelated_axis, detected_peaks, NEON_PEAKS_NM)[0]
        array([543.73052774, 543.78567712, 543.84082651, ..., 656.51102082,
               656.5661702 , 656.62131959])
    """
    polynomial, fitness = Polynomial.fit(  # type: ignore
        x_values_observed, x_values_groundtruth, deg=degree, full=True
    )
    return polynomial(x_range), fitness  # type: ignore


def configure_logging(log_filepath: str | Path | None = None) -> None:
    """Configure logging for CLI."""
    # configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s in %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if log_filepath is not None:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
