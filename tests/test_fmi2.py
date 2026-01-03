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
    "reference_fmu, expected_fmi_version",
    [
        ("2.0/Feedthrough.fmu", "2.0"),
        ("2.0/BouncingBall.fmu", "2.0"),
        ("2.0/VanDerPol.fmu", "2.0"),
        ("2.0/Dahlquist.fmu", "2.0"),
        ("2.0/Stair.fmu", "2.0"),
        ("2.0/Resource.fmu", "2.0"),
    ],
)
def test_fmi_version(reference_fmu, expected_fmi_version, reference_fmus_dir):
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)
    assert md.fmi_version == expected_fmi_version


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


# @pytest.mark.parametrize("xml_file", XML_FILES, ids=[f.name for f in XML_FILES])
# def test_parse_xml_file(xml_file):
#     """Test parsing a single XML file"""
#     model = read_model_description(xml_file)

#     # Basic assertions
#     assert model.model_name
#     assert model.fmi_version
#     assert model.model_variables

#     # Validate the model using Pydantic's validation
#     # This ensures the model can be dumped and re-validated
#     model.model_validate(model.model_dump())


# @pytest.mark.parametrize("xml_file", XML_FILES, ids=[f.name for f in XML_FILES])
# def test_model_serialization(xml_file):
#     """Test serializing the model back to dict"""
#     model = read_model_description(xml_file)

#     # Convert back to dict
#     model_dict = model.model_dump()

#     # Check consistency
#     assert model_dict["fmi_version"] == model.fmi_version
#     assert model_dict["model_name"] == model.model_name


# @pytest.mark.parametrize("xml_file", XML_FILES, ids=[f.name for f in XML_FILES])
# def test_model_details(xml_file):
#     """Test detailed properties of the model"""
#     model = read_model_description(xml_file)

#     # Check required attributes
#     assert model.guid is not None
#     assert len(model.guid) > 0
#     assert model.generation_tool is not None

#     # Check Model Variables
#     assert model.model_variables is not None
#     assert len(model.model_variables.variables) > 0

#     # Check that every variable has a name and value reference
#     for var in model.model_variables.variables:
#         assert var.name is not None
#         assert var.value_reference is not None
#         # Check that one of the types is set (Real, Integer, Boolean, String, Enumeration)
#         assert any(
#             [
#                 var.real is not None,
#                 var.integer is not None,
#                 var.boolean is not None,
#                 var.string is not None,
#                 var.enumeration is not None,
#             ]
#         )

#     # Check Model Structure if present
#     # if model.model_structure:
#     #     if model.model_structure.outputs:
#     #         for output in model.model_structure.outputs:
#     #             assert output.index > 0

#     #     if model.model_structure.derivatives:

#     # Specific checks for BouncingBall (test1.xml)
#     if model.model_name == "BouncingBall":
#         # Check specific variables exist
#         var_names = [v.name for v in model.model_variables.variables]
#         assert "time" in var_names
#         assert "h" in var_names
#         assert "v" in var_names

#         # Check unit definitions
#         assert model.unit_definitions is not None
#         unit_names = [u.name for u in model.unit_definitions.units]
#         assert "m" in unit_names
#         assert "m/s" in unit_names
