from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .typing import FloatArray

logger = logging.getLogger(__name__)


def read_openraman_csv(csv_filepath: Path | str) -> FloatArray:
    """Read data from a CSV file output by the OpenRAMAN.

    Unlike the other readers as it does not return wavenumbers or metadata. The spectral range is
    not automatically calibrated by the instrument, and thus must be calibrated in a subsequent
    step using known reference values from a standard sample. Metadata must also be captured
    separately as it is not stored in the CSV file.
    """
    dataframe = pd.read_csv(csv_filepath)  # type: ignore
    intensities = np.array(dataframe["Intensity (a.u.)"].values).astype(np.float64)
    return intensities


def read_horiba_txt(txt_filepath: Path | str) -> tuple[FloatArray, FloatArray, dict]:
    """Read data from a text file output by the Horiba MacroRam.

    Spectral range is automatically calibrated by the instrument but is in descending order
    (high wavenumbers --> low wavenumbers), so we flip the order.
    """
    # read metadata -- first 32 lines
    metadata = {}
    with open(txt_filepath, encoding="utf-8", errors="ignore") as txt_file:
        metadata_text = txt_file.read()
        metadata_lines = metadata_text.splitlines()[:32]
    for line in metadata_lines:
        property, value = line.split("=")
        property = property.replace("#", "")
        metadata[property] = value.replace("\t", "")

    # read spectral data -- starts on line 33
    column_names = ["wavenumber_cm-1", "intensity"]
    dataframe = pd.read_csv(  # type: ignore
        txt_filepath,
        skiprows=32,
        header=None,
        names=column_names,
        sep="\t",
        encoding="unicode_escape",
    )

    # convert spectral data to numpy arrays
    wavenumbers_cm1 = np.array(dataframe[column_names[0]].values)[::-1].astype(np.float64)
    intensities = np.array(dataframe[column_names[1]].values)[::-1].astype(np.float64)
    return wavenumbers_cm1, intensities, metadata


def read_renishaw_csv(csv_filepath: Path | str) -> tuple[FloatArray, FloatArray]:
    """Read data from a CSV file output by the Renishaw Qontor.

    The Renishaw Qontor stores spectral data in a sensical, but somewhat cumbersome manner
    when data is recorded using the mapping capabilities. Spectra from each (X, Y) position
    in the scan are saved and output to the same TXT file. So depending on how the experiment
    is set up, one output file could contain either multiple technical replicates, biological
    replicates, or different specimen altogether without any real way of knowing. Additionally,
    no metadata is saved. This means the raw data output by the Renishaw is usually processed and
    organized to better match the CSV files from the other spectrometers. This isn't ideal, but
    was the most practical workaround I could come up with. Example raw data:

        #X		#Y		#Wave		#Intensity
        -143.855707	-3313.582825	1808.186523	210291.000000
        -143.855707	-3313.582825	1807.220703	211172.062500
        -143.855707	-3313.582825	1806.254883	211110.218750
        ...
        -141.829040	-3327.361151	1808.186523	231103.140625
        -141.829040	-3327.361151	1807.220703	230652.421875
        -141.829040	-3327.361151	1806.254883	230639.968750
        ...

    After processing:
        `sample-1.csv` -- first (X, Y) position:
            #Wave,#Intensity
            1808.186523,210291.000000
            1807.220703,211172.062500
            1806.254883,211110.218750
            ...
        `sample-2.csv` -- next (X, Y) position:
            #Wave,#Intensity
            1808.186523,231103.140625
            1807.220703,230652.421875
            1806.254883,230639.968750
            ...

    Spectral range is automatically calibrated by the instrument but is in descending order
    (high wavenumbers --> low wavenumbers), so we flip the order.
    """
    # read spectral data -- no metadata in the header
    column_names = ["#Wave", "#Intensity"]
    dataframe = pd.read_csv(  # type: ignore
        csv_filepath,
    )

    # convert spectral data to numpy arrays
    wavenumbers_cm1 = np.array(dataframe[column_names[0]].values)[::-1].astype(np.float64)
    intensities = np.array(dataframe[column_names[1]].values)[::-1].astype(np.float64)
    return wavenumbers_cm1, intensities


def read_wasatch_csv(csv_filepath: Path | str) -> tuple[FloatArray, FloatArray, dict]:
    """Read data from a CSV file output by the WP 785X Raman Spectrometer.

    Spectral range is automatically calibrated by the instrument.
    """
    # read metadata -- first 53 lines of the CSV file
    with open(csv_filepath) as csv_file:
        metadata_text = csv_file.read()
    metadata_lines = metadata_text.splitlines()[:53]
    metadata = {line.split(",")[0]: line.split(",")[1] for line in metadata_lines}

    # read spectral data -- starts on line 55
    column_names = ["Wavelength", "Wavenumber", "Processed"]
    dataframe = pd.read_csv(  # type: ignore
        csv_filepath,
        skiprows=54,
        usecols=column_names,
    ).dropna()

    # convert spectral data to numpy arrays
    wavenumbers_cm1 = np.array(dataframe["Wavenumber"]).astype(np.float64)
    intensities = np.array(dataframe["Processed"]).astype(np.float64)

    return wavenumbers_cm1, intensities, metadata
