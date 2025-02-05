# ramanalysis

This repository contains a Python package called `ramanalysis`, the main purpose of which is to facilitate reading Raman spectroscopy data from a variety of instruments by unifying the way the spectral data is loaded. This package currently supports loading spectral data from four different Raman spectrometer manufacturers:
- Horiba (tested on the [MacroRAM](https://www.horiba.com/usa/scientific/products/detail/action/show/Product/macroramtm-805/) and [LabRAM HR Evolution](https://www.horiba.com/usa/scientific/products/detail/action/show/Product/labram-hr-evolution-1083/))
- [OpenRAMAN](https://www.open-raman.org/)
- Renishaw (tested on the [inVia Qontor](https://www.renishaw.com/en/invia-confocal-raman-microscope--6260?srsltid=AfmBOopl_QZvOFjalleTRHwKhOeAd4n04PBzR-F76UfXmM1ld-RRMILm))
- Wasatch (tested on the [WP785X](https://wasatchphotonics.com/product/wp-785x-raman-spectrometer-series/))

This package also facilitates the calibration of spectral data output by the OpenRAMAN. The calibration procedure consists of two steps:
1. A rough calibration based on a broadband excitation light source (e.g. Neon lamp)
2. A fine calibration based on Raman-scattered light from a standard sample
   (e.g. acetonitrile)

Both calibration steps can be run automatically when spectra of neon and acetonitrile are provided (see [Usage](usage)). This automated procedure builds upon the [calibration procedure](https://github.com/Arcadia-Science/2024-open-raman-analysis/blob/calibration/notebooks/0_generate_calibration.ipynb) implemented by [@sunandascript](https://github.com/sunandascript). For more information on the calibration procedure, see this [blog post](https://www.open-raman.org/robust-calibration-method-for-spectrometers/) by the creator of the OpenRAMAN.


## Installation

<!-- Hopefully possible in the near future...
The package is hosted on PyPI and can be installed using pip:

```bash
pip install ramanalysis
``` -->

The package can be installed directly from the GitHub repository via `pip`:
```bash
pip install git+https://github.com/Arcadia-Science/ramanalysis.git
```


## Usage

Read and calibrate spectral data from an OpenRAMAN CSV file.
```python
from pathlib import Path
from ramanalysis import RamanSpectrum

# Set file paths to the CSV files for your sample and calibration data
example_data_directory = Path("ramanalysis/ramanalysis/tests/example_data/OpenRAMAN/")
csv_filepath_sample = next(example_data_directory.glob("*CC-125*.csv"))
csv_filepath_excitation_calibration = next(example_data_directory.glob("*neon*.csv"))
csv_filepath_emission_calibration = next(example_data_directory.glob("*aceto*.csv"))

# Read and calibrate the spectral data from your sample
spectrum = RamanSpectrum.from_openraman_csvfiles(
    csv_filepath_sample,
    csv_filepath_excitation_calibration,
    csv_filepath_emission_calibration,
)
```

See [examples](docs/examples/) for more example usage.


## Roadmap
1. Add a reader for CRS data from Leica LIF files using [`readlif`](https://github.com/Arcadia-Science/readlif).
2. Integrate with [`RamanSPy`](https://ramanspy.readthedocs.io/en/latest/index.html) to easily convert `RamanSpectrum` instances to `ramanspy` `Spectrum` or `SpectralImage` instances and vice versa. Would look something like this:
   ```python
   spectrum = RamanSpectrum.from_openraman_csvfiles(
        csv_filepath_sample,
        csv_filepath_excitation_calibration,
        csv_filepath_emission_calibration,
    )

   ramanspy_spectrum = spectrum.to_ramanspy_spectrum()
   ramanalysis_spectrum = RamanSpectrum.from_ramanspy_spectrum(ramanspy_spectrum)
   ```


## Contributing

If you are interested in contributing to this package, please check out the [developer notes](docs/development.md).
See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
