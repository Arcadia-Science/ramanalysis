import logging
from pathlib import Path

from numpy.polynomial import Polynomial

from .typing import FloatArray, ScalarArray

logger = logging.getLogger(__name__)


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
