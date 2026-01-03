# mdreader 📖

A Python library for reading and parsing Functional Mock-up Interface (FMI2.0 currently) model description XML file into Pydantic models.

[![PyPI version](https://badge.fury.io/py/mdreader.svg)](https://badge.fury.io/py/mdreader)

## Installation 📦

Add `mdreader` to your `pyproject.toml` with `uv` with:

``` bash
uv add mdreader
```

> To install `uv`, see https://docs.astral.sh/uv/getting-started/installation/

## How to use 🚀

To read and parse an FMI 2.0 model description:

```python
from mdreader.fmi2 import read_model_description

# Read from XML file
md = read_model_description("path/to/modelDescription.xml")

# Read from FMU archive
md = read_model_description("path/to/model.fmu")

# Read from unzipped FMU directory
md = read_model_description("path/to/unzipped/fmu/directory")

print(md)
```

## Features ✨

- Parse FMI 2.0 model description XML files
- Read model information from FMU archives
- Access model metadata (name, version, author, GUID, etc.)
- Extract variable definitions (real, integer, boolean, string, enumeration)
- Access unit definitions and type definitions
- Support for both Model Exchange and Co-Simulation interfaces
- Parse model structure, dependencies, and experiment configurations

## Why another FMI model description reader? 🤔

* **Lightweight**: mdreader only depends on Pydantic
* **De/Serialization**: Pydantic models support easy serialization to/from JSON, dict, etc.
* **Validation**: mdreader uses Pydantic models to ensure the integrity of the parsed data
* **FMI version exclusive**: The `fmi2.FMIModelDescription` class is specific to FMI 2.0 (not a mix of FMI1/2/3), making it simpler to use for that version

## Related projects 🔗

* [fmpy](https://github.com/CATIA-Systems/FMPy): A similar `read_model_description` function is available in FMPy, but it uses custom classes instead of Pydantic models and has more dependencies.

## Licensing 📄

The code in this project is licensed under MIT license.
See the [LICENSE](LICENSE) file for details.
