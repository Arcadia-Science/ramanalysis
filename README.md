# ramanalysis

This repository contains a Python package called `ramanalysis`, the main purpose of which is to facilitate reading Raman spectroscopy data from a variety of instruments by unifying the way the spectral data is loaded.

<!-- Currently, this package supports loading spectral data from two different Raman spectrometers:
- [OpenRAMAN](https://www.open-raman.org/)
- [Horiba MacroRam](https://www.horiba.com/usa/scientific/products/detail/action/show/Product/macroramtm-805/) -->


## Installation

<!-- Hopefully possible in the near future...
The package is hosted on PyPI and can be installed using pip:

```bash
pip install ramanalysis
``` -->

Clone the repository and install via pip:
```bash
git clone https://github.com/Arcadia-Science/ramanalysis.git
cd ramanalysis
pip install -e .
```


## Usage

Read and calibrate spectral data from an OpenRAMAN CSV file.
```python
from ramanalysis import RamanSpectrum

# Set file paths to the CSV files for your sample and calibration data
example_data_directory = Path("../../ramanalysis/tests/example_data")
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

See [examples](examples/) for more example usage.


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
