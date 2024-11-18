from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .typing import FloatArray

logger = logging.getLogger(__name__)


def read_openraman_csv(csv_filepath: Path | str) -> FloatArray:
    """Read data from a CSV file output by the OpenRAMAN.

    Spectral range is not automatically calibrated by the instrument, and thus must be calibrated
    in a subsequent step using known reference values from a standard sample.
    """
    dataframe = pd.read_csv(csv_filepath)  # type: ignore
    return np.array(dataframe["Intensity (a.u.)"].values)


def read_horiba_txt(txt_filepath: Path | str) -> tuple[FloatArray, FloatArray]:
    """Read data from a text file output by the Horiba MacroRam.

    Spectral range is automatically calibrated by the instrument but is in descending order
    (high wavenumbers --> low wavenumbers), so we flip the order around.
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
    wavenumbers_cm1 = np.array(dataframe["wavenumber_cm-1"].values)[::-1]
    intensities = np.array(dataframe["intensity"].values)[::-1]
    return wavenumbers_cm1, intensities
