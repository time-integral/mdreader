import pathlib
import zipfile

import httpx
import pytest

from mdreader.fmi2 import read_model_description

REFERENCE_FMUS_URL = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.38/Reference-FMUs-0.0.38.zip"


@pytest.fixture(scope="session")
def reference_fmus_dir(tmp_path_factory):
    """Download and extract Reference-FMUs once per test session."""
    tmpdir = tmp_path_factory.mktemp("reference_fmus")

    # Download the reference FMU zip file
    response = httpx.get(REFERENCE_FMUS_URL, follow_redirects=True)
    response.raise_for_status()

    zip_path = tmpdir / "Reference-FMUs.zip"
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir)

    return tmpdir


@pytest.mark.parametrize(
    "reference_fmu, expected_metadata",
    [
        (
            "2.0/Feedthrough.fmu",
            {
                "fmi_version": "2.0",
                "model_name": "Feedthrough",
                "guid": "{37B954F1-CC86-4D8F-B97F-C7C36F6670D2}",
                "description": "A model to test different variable types, causalities, and variabilities",
                "author": None,
                "version": None,
                "copyright": None,
                "license": None,
                "generation_tool": "Reference FMUs (v0.0.38)",
                "generation_date_and_time": "2025-02-04T04:14:59.281054+00:00",
            },
        ),
        (
            "2.0/BouncingBall.fmu",
            {
                "fmi_version": "2.0",
                "model_name": "BouncingBall",
                "guid": "{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
                "description": "This model calculates the trajectory, over time, of a ball dropped from a height of 1 m",
                "author": None,
                "version": None,
                "copyright": None,
                "license": None,
                "generation_tool": "Reference FMUs (v0.0.38)",
                "generation_date_and_time": "2025-02-04T04:14:58.875645+00:00",
            },
        ),
        (
            "2.0/VanDerPol.fmu",
            {
                "fmi_version": "2.0",
                "model_name": "Van der Pol oscillator",
                "guid": "{BD403596-3166-4232-ABC2-132BDF73E644}",
                "description": "This model implements the van der Pol oscillator",
                "author": None,
                "version": None,
                "copyright": None,
                "license": None,
                "generation_tool": "Reference FMUs (v0.0.38)",
                "generation_date_and_time": "2025-02-04T04:14:59.010953+00:00",
            },
        ),
        (
            "2.0/Dahlquist.fmu",
            {
                "fmi_version": "2.0",
                "model_name": "Dahlquist",
                "guid": "{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
                "description": "This model implements the Dahlquist test equation",
                "author": None,
                "version": None,
                "copyright": None,
                "license": None,
                "generation_tool": "Reference FMUs (v0.0.38)",
                "generation_date_and_time": "2025-02-04T04:14:59.489702+00:00",
            },
        ),
        (
            "2.0/Stair.fmu",
            {
                "fmi_version": "2.0",
                "model_name": "Stair",
                "guid": "{BD403596-3166-4232-ABC2-132BDF73E644}",
                "description": "This model generates a stair signal using time events",
                "author": None,
                "version": None,
                "copyright": None,
                "license": None,
                "generation_tool": "Reference FMUs (v0.0.38)",
                "generation_date_and_time": "2025-02-04T04:14:59.396420+00:00",
            },
        ),
        (
            "2.0/Resource.fmu",
            {
                "fmi_version": "2.0",
                "model_name": "Resource",
                "guid": "{7b9c2114-2ce5-4076-a138-2cbc69e069e5}",
                "description": "This model loads data from a resource file",
                "author": None,
                "version": None,
                "copyright": None,
                "license": None,
                "generation_tool": "Reference FMUs (v0.0.38)",
                "generation_date_and_time": "2025-02-04T04:14:59.098389+00:00",
            },
        ),
    ],
)
def test_metadata(reference_fmu, expected_metadata, reference_fmus_dir):
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)
    assert md.fmi_version == expected_metadata["fmi_version"]
    assert md.model_name == expected_metadata["model_name"]
    assert md.guid == expected_metadata["guid"]
    assert md.description == expected_metadata["description"]
    assert md.author == expected_metadata["author"]
    assert md.version == expected_metadata["version"]
    assert md.copyright == expected_metadata["copyright"]
    assert md.license == expected_metadata["license"]
    assert md.generation_tool == expected_metadata["generation_tool"]
    assert md.generation_date_and_time == expected_metadata["generation_date_and_time"]


@pytest.mark.parametrize(
    "reference_fmu, expected_inputs, expected_outputs",
    [
        (
            "2.0/Feedthrough.fmu",
            [
                "Float64_continuous_input",
                "Float64_discrete_input",
                "Int32_input",
                "Boolean_input",
                "Enumeration_input",
            ],
            [
                "Float64_continuous_output",
                "Float64_discrete_output",
                "Int32_output",
                "Boolean_output",
                "Enumeration_output",
            ],
        ),
        ("2.0/BouncingBall.fmu", [], ["h", "v"]),
        ("2.0/VanDerPol.fmu", [], ["x0", "x1"]),
        ("2.0/Dahlquist.fmu", [], ["x"]),
        ("2.0/Stair.fmu", [], ["counter"]),
        ("2.0/Resource.fmu", [], ["y"]),
    ],
)
def test_scarlar_variables(
    reference_fmu, expected_inputs, expected_outputs, reference_fmus_dir
):
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)
    input_vars = [
        var.name for var in md.model_variables.variables if var.causality == "input"
    ]
    output_vars = [
        var.name for var in md.model_variables.variables if var.causality == "output"
    ]
    print("Input Vars:", input_vars)
    print("Output Vars:", output_vars)
    assert sorted(input_vars) == sorted(expected_inputs)
    assert sorted(output_vars) == sorted(expected_outputs)
