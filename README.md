# ramanalysis

This repo contains a Python package called `ramanalysis`, the main purpose of which is to facilitate processing Raman spectroscopy data.

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

Read an OpenRAMAN CSV file.
```python
from ramanalysis import RamanSpectrum

example_data_directory = Path("../../ramanalysis/tests/example_data")

csv_filepath_sample = next(example_data_directory.glob("*CC-125*.csv"))
csv_filepath_excitation_calibration = next(example_data_directory.glob("*neon*.csv"))
csv_filepath_emission_calibration = next(example_data_directory.glob("*aceto*.csv"))

spectrum = RamanSpectrum.from_openraman_csvfiles(
    csv_filepath_sample,
    csv_filepath_excitation_calibration,
    csv_filepath_emission_calibration,
)
```


## Contributing

If you are interested in contributing to this package, please check out the [developer notes](docs/development.md).
See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
