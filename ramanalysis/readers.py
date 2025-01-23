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

    As the software and hardware for the OpenRAMAN is under active development, the format of the
    output CSV file is not static. Sometimes the column name is "Intensity (a.u.)" and sometimes
    it is "Intensity".
    """
    dataframe = pd.read_csv(csv_filepath)  # type: ignore

    valid_intensity_column_names = {"Intensity", "Intensity (a.u.)"}
    for column_name in dataframe.columns:
        if column_name in valid_intensity_column_names:
            intensity_column_name = column_name
            break
    else:
        raise KeyError(
            f"Expected one of {valid_intensity_column_names}, but received "
            f"these columns instead: {dataframe.columns.tolist()}."
        )

    intensities = np.array(dataframe.loc[:, intensity_column_name].values).astype(np.float64)
    return intensities


def read_horiba_txt(
    txt_filepath: Path | str,
    num_skip_rows: int = 32,
) -> tuple[FloatArray, FloatArray, dict]:
    """Read data from a text file output by the Horiba MacroRam or the Horiba LabRAM.

    The TXT files output by the two different instruments vary only in the number of rows in
    the header: 32 for the MacroRAM and 47 for the LabRAM. Spectral range is automatically
    calibrated by the instrument but is in descending order (high wavenumbers --> low wavenumbers),
    so we flip the order.
    """
    # read metadata
    metadata = {}
    with open(txt_filepath, encoding="utf-8", errors="ignore") as txt_file:
        metadata_text = txt_file.read()
        metadata_lines = metadata_text.splitlines()[:num_skip_rows]
    for line in metadata_lines:
        property, value = line.split("=")
        property = property.replace("#", "")
        metadata[property] = value.replace("\t", "")

    # read spectral data
    column_names = ["wavenumber_cm-1", "intensity"]
    dataframe = pd.read_csv(  # type: ignore
        txt_filepath,
        skiprows=num_skip_rows,
        header=None,
        names=column_names,
        sep="\t",
        encoding="unicode_escape",
    )

    # convert spectral data to numpy arrays
    wavenumbers_cm1 = np.array(dataframe[column_names[0]].values)[::-1].astype(np.float64)
    intensities = np.array(dataframe[column_names[1]].values)[::-1].astype(np.float64)
    return wavenumbers_cm1, intensities, metadata


def read_renishaw_singlepoint_txt(txt_filepath: Path | str) -> tuple[FloatArray, FloatArray]:
    """Read data from a TXT file output by the Renishaw Qontor single-point scan.

    Data from the Renishaw comes in (at least) two slightly different forms: single-point scanning
    and multipoint scanning. Example raw data from a single-point scan:

        #Wave       #Intensity
        3399.408203 1016.205444
        3398.738281 1008.868225
        3398.067383 1016.025818
        ...
        602.157227  41879.410156
        600.934570  41363.480469

    See also:
        - To read data from a multipoint scan, see :func:`read_renishaw_multipoint_txt`
    """
    # read spectral data -- no metadata in the header
    dataframe = pd.read_csv(  # type: ignore
        txt_filepath,
        sep=r"\s+",
    )

    # check that file has expected column names
    singlepoint_column_names = ["#Wave", "#Intensity"]
    if len(dataframe.columns) != 2 or not all(
        column in dataframe.columns for column in singlepoint_column_names
    ):
        msg = (
            "Expected columns '#Wave' and '#Intensity', but received "
            f"{dataframe.columns.tolist()} instead."
        )
        raise KeyError(msg)

    # convert spectral data to numpy arrays
    wavenumbers_cm1 = np.array(dataframe["#Wave"].values)[::-1].astype(np.float64)
    intensities = np.array(dataframe["#Intensity"].values)[::-1].astype(np.float64)
    return wavenumbers_cm1, intensities


def read_renishaw_multipoint_txt(
    txt_filepath: Path | str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Read data from a TXT file output by the Renishaw Qontor multipoint scan.

    Data from the Renishaw comes in (at least) two slightly different forms: single-point scanning
    and multipoint scanning. Multipoint scanning can either be in a grid-like pattern to form
    an image or just an arbitrary sequence of points. Example raw data from a multipoint scan:

        #X          #Y          #Wave       #Intensity
        -8.312373   145.838849  1808.186523 221543.765625
        -8.312373   145.838849  1807.220703 222801.703125
        -8.312373   145.838849  1806.254883 222370.828125
        ...
        66.420960   84.856999   713.626953  118237.062500
        66.420960   84.856999   712.416016  118192.578125

    There is no metadata saved in the TXT file to read. Spectral range is automatically calibrated
    by the instrument but is in descending order (high wavenumbers --> low wavenumbers), so we
    flip the order. Wavenumber data is redundant in that each point in the scan has intensity
    values distributed over the same wavenumber range.

    See also:
        - To read data from a single-point scan, see :func:`read_renishaw_singlepoint_txt`.
    """
    # read spectral data -- no metadata in the header
    dataframe = pd.read_csv(  # type: ignore
        txt_filepath,
        sep=r"\s+",
    )

    # check that file has expected column names
    multipoint_column_names = ["#X", "#Y", "#Wave", "#Intensity"]
    if len(dataframe.columns) != 4 or not all(
        column in dataframe.columns for column in multipoint_column_names
    ):
        msg = (
            "Expected columns '#X', '#Y', '#Wave' and '#Intensity', but received "
            f"{dataframe.columns.tolist()} instead."
        )
        raise KeyError(msg)

    # extract wavenumber and intensity data at each (X, Y) position
    x_positions: list[float] = []
    y_positions: list[float] = []
    intensities = []
    for (x_position, y_position), spectral_data in dataframe.groupby(["#X", "#Y"], sort=False):  # type: ignore
        wavenumbers_cm1 = spectral_data["#Wave"].values[::-1]
        intensities.append(spectral_data["#Intensity"].values[::-1])  # type: ignore
        x_positions.append(x_position)
        y_positions.append(y_position)

    # convert spectral data and position data to numpy arrays
    wavenumbers_cm1 = np.array(wavenumbers_cm1, dtype=np.float64)
    intensities = np.array(intensities, dtype=np.float64)
    positions = np.stack([x_positions, y_positions], axis=1)

    return wavenumbers_cm1, intensities, positions


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
