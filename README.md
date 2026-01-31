# mdreader 📖

A Python library for reading and parsing Functional Mock-up Interface (FMI 2.0 and 3.0) model description XML files into Pydantic models.

[![PyPI version](https://badge.fury.io/py/mdreader.svg)](https://badge.fury.io/py/mdreader)

## Installation 📦

Add `mdreader` to your `pyproject.toml` with `uv` with:

``` bash
uv add mdreader
```

> To install `uv`, see https://docs.astral.sh/uv/getting-started/installation/

## How to use 🚀

To read and parse an FMI model description (works for both 2.0 and 3.0):

```python
import mdreader.fmi2 as fmi2
import mdreader.fmi3 as fmi3

# Read from XML file
md = fmi2.read_model_description("path/to/fmi2/modelDescription.xml")  # or fmi3.read_model_description for 3.0
# Read from FMU archive
md = fmi2.read_model_description("path/to/model.fmu")  # or fmi3.read_model_description for 3.0

# Read from unzipped FMU directory
md = fmi2.read_model_description("path/to/unzipped/fmu/directory")  # or fmi3.read_model_description for 3.0

print(md)
```

## Features ✨

- Parse FMI 2.0 and 3.0 model description XML files
- Read model information from FMU archives
- Access model metadata (name, version, author, GUID, etc.)
- Extract variable definitions (real, integer, boolean, string, enumeration, clock, binary)
- Access unit definitions and type definitions
- Support for Model Exchange, Co-Simulation, and Scheduled Execution interfaces
- Parse model structure, dependencies, and experiment configurations
- Full support for FMI 3.0 features including structural parameters, clock variables, and directional derivatives

## Why another FMI model description reader? 🤔

* **Lightweight**: mdreader only depends on Pydantic
* **De/Serialization**: Pydantic models support easy serialization to/from JSON, dict, etc.
* **Validation**: mdreader uses Pydantic models to ensure the integrity of the parsed data
* **FMI version specific**: The `fmi2.FMIModelDescription` and `fmi3.FMIModelDescription` classes are specific to their respective FMI versions (not a mix of versions), making it simpler to use for each version

## Related projects 🔗

* [fmpy](https://github.com/CATIA-Systems/FMPy): A similar `read_model_description` function is available in FMPy, but it uses custom classes instead of Pydantic models and has more dependencies.

## Licensing 📄

The code in this project is licensed under MIT license.
See the [LICENSE](LICENSE) file for details.
