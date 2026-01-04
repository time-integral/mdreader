import pathlib
import zipfile

import httpx
import pytest

from mdreader.fmi2 import read_model_description

REFERENCE_FMUS_URL = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.39/Reference-FMUs-0.0.39.zip"


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
                "generation_tool": "Reference FMUs (v0.0.39)",
                "number_of_event_indicators": 0,
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
                "generation_tool": "Reference FMUs (v0.0.39)",
                "number_of_event_indicators": 1,
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
                "generation_tool": "Reference FMUs (v0.0.39)",
                "number_of_event_indicators": 0,
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
                "generation_tool": "Reference FMUs (v0.0.39)",
                "number_of_event_indicators": 0,
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
                "generation_tool": "Reference FMUs (v0.0.39)",
                "number_of_event_indicators": 0,
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
                "generation_tool": "Reference FMUs (v0.0.39)",
                "number_of_event_indicators": 0,
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
    assert (
        md.number_of_event_indicators == expected_metadata["number_of_event_indicators"]
    )


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
                "String_input",
            ],
            [
                "Float64_continuous_output",
                "Float64_discrete_output",
                "Int32_output",
                "Boolean_output",
                "Enumeration_output",
                "String_output",
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
    assert sorted(input_vars) == sorted(expected_inputs)
    assert sorted(output_vars) == sorted(expected_outputs)


@pytest.mark.parametrize(
    "reference_fmu, expected_units",
    [
        (
            "2.0/BouncingBall.fmu",
            [
                {"name": "m", "base_unit": {"m": 1}},
                {"name": "m/s", "base_unit": {"m": 1, "s": -1}},
                {"name": "m/s2", "base_unit": {"m": 1, "s": -2}},
            ],
        ),
        # Feedthrough and others may not have unit definitions
        ("2.0/Feedthrough.fmu", []),
        ("2.0/VanDerPol.fmu", []),
        ("2.0/Dahlquist.fmu", []),
        ("2.0/Stair.fmu", []),
        ("2.0/Resource.fmu", []),
    ],
)
def test_unit_definitions(reference_fmu, expected_units, reference_fmus_dir):
    """Test that unit definitions are properly parsed with correct values"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if expected_units:
        assert md.unit_definitions is not None
        assert len(md.unit_definitions.units) == len(expected_units)

        for i, expected_unit in enumerate(expected_units):
            unit = md.unit_definitions.units[i]
            assert unit.name == expected_unit["name"]

            if "base_unit" in expected_unit:
                base_unit = unit.base_unit
                expected_base = expected_unit["base_unit"]
                for attr, expected_val in expected_base.items():
                    assert getattr(base_unit, attr) == expected_val
    else:
        # For FMUs without unit definitions, ensure it's None or empty
        if md.unit_definitions:
            assert len(md.unit_definitions.units) == 0


@pytest.mark.parametrize(
    "reference_fmu, expected_types",
    [
        (
            "2.0/Feedthrough.fmu",
            [
                {
                    "name": "Option",
                    "type_category": "Enumeration",
                    "items": [
                        {"name": "Option 1", "value": 1, "description": "First option"},
                        {
                            "name": "Option 2",
                            "value": 2,
                            "description": "Second option",
                        },
                    ],
                }
            ],
        ),
        (
            "2.0/BouncingBall.fmu",
            [
                {
                    "name": "Position",
                    "type_category": "Real",
                    "quantity": "Position",
                    "unit": "m",
                },
                {
                    "name": "Velocity",
                    "type_category": "Real",
                    "quantity": "Velocity",
                    "unit": "m/s",
                },
                {
                    "name": "Acceleration",
                    "type_category": "Real",
                    "quantity": "Acceleration",
                    "unit": "m/s2",
                },
            ],
        ),
        # Other FMUs may not have type definitions
        ("2.0/VanDerPol.fmu", []),
        ("2.0/Dahlquist.fmu", []),
        ("2.0/Stair.fmu", []),
        ("2.0/Resource.fmu", []),
    ],
)
def test_type_definitions(reference_fmu, expected_types, reference_fmus_dir):
    """Test that type definitions are properly parsed with correct values"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if expected_types:
        assert md.type_definitions is not None
        assert len(md.type_definitions.simple_types) == len(expected_types)

        for i, expected_type in enumerate(expected_types):
            simple_type = md.type_definitions.simple_types[i]
            assert simple_type.name == expected_type["name"]
            assert simple_type.get_type_category() == expected_type["type_category"]

            if "items" in expected_type and simple_type.enumeration:
                for j, expected_item in enumerate(expected_type["items"]):
                    item = simple_type.enumeration.items[j]
                    assert item.name == expected_item["name"]
                    assert item.value == expected_item["value"]
                    assert item.description == expected_item["description"]
            elif "quantity" in expected_type:
                if simple_type.real:
                    assert simple_type.real.quantity == expected_type["quantity"]
                if "unit" in expected_type:
                    assert simple_type.real.unit == expected_type["unit"]
    else:
        # For FMUs without type definitions, ensure it's None or empty
        if md.type_definitions:
            assert len(md.type_definitions.simple_types) == 0


@pytest.mark.parametrize(
    "reference_fmu, expected_variables",
    [
        # Test specific variables from Feedthrough
        (
            "2.0/Feedthrough.fmu",
            [
                {
                    "name": "time",
                    "value_reference": 0,
                    "causality": "independent",
                    "variability": "continuous",
                    "type": "Real",
                },
                {
                    "name": "Float64_continuous_input",
                    "value_reference": 7,
                    "causality": "input",
                    "variability": "continuous",
                    "type": "Real",
                    "start": 0.0,
                },
                {
                    "name": "Int32_input",
                    "value_reference": 19,
                    "causality": "input",
                    "variability": "discrete",
                    "type": "Integer",
                    "start": 0,
                },
                {
                    "name": "Boolean_input",
                    "value_reference": 27,
                    "causality": "input",
                    "variability": "discrete",
                    "type": "Boolean",
                    "start": False,
                },
                {
                    "name": "String_input",
                    "value_reference": 29,
                    "causality": "input",
                    "variability": "discrete",
                    "type": "String",
                    "start": "Set me!",
                },
                {
                    "name": "Enumeration_input",
                    "value_reference": 33,
                    "causality": "input",
                    "variability": "discrete",
                    "type": "Enumeration",
                    "declared_type": "Option",
                    "start": 1,
                },
            ],
        ),
        # Test specific variables from BouncingBall
        (
            "2.0/BouncingBall.fmu",
            [
                {
                    "name": "time",
                    "value_reference": 0,
                    "causality": "independent",
                    "variability": "continuous",
                    "description": "Simulation time",
                    "type": "Real",
                },
                {
                    "name": "h",
                    "value_reference": 1,
                    "causality": "output",
                    "variability": "continuous",
                    "initial": "exact",
                    "description": "Position of the ball",
                    "type": "Real",
                    "start": 1.0,
                    "reinit": True,
                    "declared_type": "Position",
                },
                {
                    "name": "g",
                    "value_reference": 5,
                    "causality": "parameter",
                    "variability": "fixed",
                    "initial": "exact",
                    "description": "Gravity acting on the ball",
                    "type": "Real",
                    "start": -9.81,
                    "declared_type": "Acceleration",
                },
            ],
        ),
    ],
)
def test_variable_properties(reference_fmu, expected_variables, reference_fmus_dir):
    """Test that variable properties are correctly parsed from XML"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    # Create a mapping of variable names to variables for easy lookup
    var_map = {var.name: var for var in md.model_variables.variables}

    for expected_var in expected_variables:
        var_name = expected_var["name"]
        assert var_name in var_map, f"Variable {var_name} not found in model"

        var = var_map[var_name]

        # Check common properties
        assert var.name == expected_var["name"]
        assert var.value_reference == expected_var["value_reference"]
        assert var.causality.value == expected_var["causality"]
        assert var.variability.value == expected_var["variability"]

        if "description" in expected_var:
            assert var.description == expected_var["description"]

        if "initial" in expected_var:
            assert var.initial.value == expected_var["initial"]

        # Check variable-specific properties
        var_type = var.get_variable_type()
        assert var_type == expected_var["type"]

        if var_type == "Real":
            real_var = var.real
            if "start" in expected_var:
                assert real_var.start == expected_var["start"]
            if "declared_type" in expected_var:
                assert real_var.declared_type == expected_var["declared_type"]
            if "reinit" in expected_var:
                assert real_var.reinit == expected_var["reinit"]
        elif var_type == "Integer":
            integer_var = var.integer
            if "start" in expected_var:
                assert integer_var.start == expected_var["start"]
        elif var_type == "Boolean":
            boolean_var = var.boolean
            if "start" in expected_var:
                assert boolean_var.start == expected_var["start"]
        elif var_type == "String":
            string_var = var.string
            if "start" in expected_var:
                assert string_var.start == expected_var["start"]
        elif var_type == "Enumeration":
            enumeration_var = var.enumeration
            if "declared_type" in expected_var:
                assert enumeration_var.declared_type == expected_var["declared_type"]
            if "start" in expected_var:
                assert enumeration_var.start == expected_var["start"]


@pytest.mark.parametrize(
    "reference_fmu, expected_model_exchange",
    [
        (
            "2.0/Feedthrough.fmu",
            {
                "model_identifier": "Feedthrough",
                "can_not_use_memory_management_functions": True,
                "can_get_and_set_fmu_state": True,
            },
        ),
        (
            "2.0/BouncingBall.fmu",
            {
                "model_identifier": "BouncingBall",
                "can_not_use_memory_management_functions": True,
                "can_get_and_set_fmu_state": True,
            },
        ),
    ],
)
def test_model_exchange_interface(
    reference_fmu, expected_model_exchange, reference_fmus_dir
):
    """Test that Model Exchange interface is correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    assert md.model_exchange is not None
    assert (
        md.model_exchange.model_identifier
        == expected_model_exchange["model_identifier"]
    )
    assert (
        md.model_exchange.can_not_use_memory_management_functions
        == expected_model_exchange["can_not_use_memory_management_functions"]
    )
    assert (
        md.model_exchange.can_get_and_set_fmu_state
        == expected_model_exchange["can_get_and_set_fmu_state"]
    )


@pytest.mark.parametrize(
    "reference_fmu, expected_co_simulation",
    [
        (
            "2.0/Feedthrough.fmu",
            {
                "model_identifier": "Feedthrough",
                "can_handle_variable_communication_step_size": True,
                "can_not_use_memory_management_functions": True,
            },
        ),
        (
            "2.0/BouncingBall.fmu",
            {
                "model_identifier": "BouncingBall",
                "can_handle_variable_communication_step_size": True,
                "can_not_use_memory_management_functions": True,
            },
        ),
    ],
)
def test_co_simulation_interface(
    reference_fmu, expected_co_simulation, reference_fmus_dir
):
    """Test that Co-Simulation interface is correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    assert md.co_simulation is not None
    assert (
        md.co_simulation.model_identifier == expected_co_simulation["model_identifier"]
    )
    assert (
        md.co_simulation.can_handle_variable_communication_step_size
        == expected_co_simulation["can_handle_variable_communication_step_size"]
    )
    assert (
        md.co_simulation.can_not_use_memory_management_functions
        == expected_co_simulation["can_not_use_memory_management_functions"]
    )


@pytest.mark.parametrize(
    "reference_fmu, expected_default_experiment",
    [
        (
            "2.0/Feedthrough.fmu",
            {"stop_time": 2.0},
        ),
        (
            "2.0/BouncingBall.fmu",
            {"start_time": 0.0, "stop_time": 3.0, "step_size": 1e-2},
        ),
        (
            "2.0/VanDerPol.fmu",
            {"start_time": 0.0, "stop_time": 20.0, "step_size": 1e-2},
        ),
    ],
)
def test_default_experiment_values(
    reference_fmu, expected_default_experiment, reference_fmus_dir
):
    """Test that DefaultExperiment values are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    assert md.default_experiment is not None

    if "start_time" in expected_default_experiment:
        assert (
            md.default_experiment.start_time
            == expected_default_experiment["start_time"]
        )
    if "stop_time" in expected_default_experiment:
        assert (
            md.default_experiment.stop_time == expected_default_experiment["stop_time"]
        )
    if "step_size" in expected_default_experiment:
        assert (
            md.default_experiment.step_size == expected_default_experiment["step_size"]
        )


@pytest.mark.parametrize(
    "reference_fmu, expected_log_categories",
    [
        (
            "2.0/Feedthrough.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
        (
            "2.0/BouncingBall.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
    ],
)
def test_log_categories(reference_fmu, expected_log_categories, reference_fmus_dir):
    """Test that log categories are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    assert md.log_categories is not None
    assert len(md.log_categories.categories) == len(expected_log_categories)

    for i, expected_category in enumerate(expected_log_categories):
        category = md.log_categories.categories[i]
        assert category.name == expected_category["name"]
        assert category.description == expected_category["description"]


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "2.0/BouncingBall.fmu",
        "2.0/VanDerPol.fmu",
    ],
)
def test_model_structure_outputs(reference_fmu, reference_fmus_dir):
    """Test that ModelStructure outputs are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if md.model_structure and md.model_structure.outputs:
        # Check that outputs exist and have correct structure
        for unknown in md.model_structure.outputs.unknowns:
            assert isinstance(unknown.index, int)
            if unknown.dependencies:
                assert all(isinstance(dep, int) for dep in unknown.dependencies)
            if unknown.dependencies_kind:
                assert all(hasattr(dep, "value") for dep in unknown.dependencies_kind)


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "2.0/BouncingBall.fmu",
        "2.0/VanDerPol.fmu",
    ],
)
def test_model_structure_derivatives(reference_fmu, reference_fmus_dir):
    """Test that ModelStructure derivatives are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if md.model_structure and md.model_structure.derivatives:
        # Check that derivatives exist and have correct structure
        for unknown in md.model_structure.derivatives.unknowns:
            assert isinstance(unknown.index, int)
            if unknown.dependencies:
                assert all(isinstance(dep, int) for dep in unknown.dependencies)
            if unknown.dependencies_kind:
                assert all(hasattr(dep, "value") for dep in unknown.dependencies_kind)


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "2.0/BouncingBall.fmu",
        "2.0/VanDerPol.fmu",
    ],
)
def test_model_structure_initial_unknowns(reference_fmu, reference_fmus_dir):
    """Test that ModelStructure initial unknowns are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if md.model_structure and md.model_structure.initial_unknowns:
        # Check that initial unknowns exist and have correct structure
        for unknown in md.model_structure.initial_unknowns.unknowns:
            assert isinstance(unknown.index, int)
            if unknown.dependencies:
                assert all(isinstance(dep, int) for dep in unknown.dependencies)
            if unknown.dependencies_kind:
                assert all(hasattr(dep, "value") for dep in unknown.dependencies_kind)


def test_variable_validation():
    """Test validation logic in RealSimpleType and IntegerSimpleType"""
    from mdreader.fmi2 import RealSimpleType, IntegerSimpleType

    # Test RealSimpleType validation - max >= min
    with pytest.raises(ValueError, match="max.*must be >= min"):
        RealSimpleType(min_value=10.0, max_value=5.0)

    # Test IntegerSimpleType validation - max >= min
    with pytest.raises(ValueError, match="max.*must be >= min"):
        IntegerSimpleType(min_value=10, max_value=5)


def test_str_to_bool_function():
    """Test the _str_to_bool helper function"""
    from mdreader.fmi2 import _str_to_bool

    # Test true values
    assert _str_to_bool("true") is True
    assert _str_to_bool("True") is True
    assert _str_to_bool("1") is True
    assert _str_to_bool("yes") is True
    assert _str_to_bool("on") is True

    # Test false values
    assert _str_to_bool("false") is False
    assert _str_to_bool("False") is False
    assert _str_to_bool("0") is False
    assert _str_to_bool("no") is False
    assert _str_to_bool("off") is False
    assert _str_to_bool("") is False

    # Test None input
    assert _str_to_bool(None) is None


def test_read_model_description_unsupported_file_type():
    """Test that read_model_description raises error for unsupported file types"""
    from mdreader.fmi2 import read_model_description
    import tempfile
    import os

    # Create a temporary file with unsupported extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"dummy content")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Unsupported file type"):
            read_model_description(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_get_type_category_methods():
    """Test the get_type_category and get_variable_type methods"""
    from mdreader.fmi2 import SimpleType, RealSimpleType, ScalarVariable, RealVariable

    # Test SimpleType.get_type_category
    real_simple_type = RealSimpleType()
    simple_type = SimpleType(name="test", real=real_simple_type)
    assert simple_type.get_type_category() == "Real"

    # Test ScalarVariable.get_variable_type
    real_var = RealVariable()
    scalar_var = ScalarVariable(name="test", value_reference=1, real=real_var)
    assert scalar_var.get_variable_type() == "Real"


def test_error_conditions_in_parsing():
    """Test error conditions in XML parsing"""
    from mdreader.fmi2 import (
        _parse_xml_to_model,
        _parse_model_exchange,
        _parse_co_simulation,
        _parse_unit,
        _parse_display_unit,
        _parse_simple_type,
        _parse_enumeration_simple_type,
    )
    import xml.etree.ElementTree as ET

    # Test missing fmiVersion attribute
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription modelName="Test" guid="test">
    </fmiModelDescription>"""

    with pytest.raises(ValueError, match="fmiVersion attribute is required"):
        _parse_xml_to_model(ET.fromstring(xml_content))

    # Test missing modelName attribute
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription fmiVersion="2.0" guid="test">
    </fmiModelDescription>"""

    with pytest.raises(ValueError, match="modelName attribute is required"):
        _parse_xml_to_model(ET.fromstring(xml_content))

    # Test missing GUID attribute
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription fmiVersion="2.0" modelName="Test">
    </fmiModelDescription>"""

    with pytest.raises(ValueError, match="GUID attribute is required"):
        _parse_xml_to_model(ET.fromstring(xml_content))

    # Test ModelExchange element without modelIdentifier
    me_xml = ET.fromstring("<ModelExchange></ModelExchange>")
    with pytest.raises(
        ValueError, match="ModelExchange element must have modelIdentifier attribute"
    ):
        _parse_model_exchange(me_xml)

    # Test CoSimulation element without modelIdentifier
    cs_xml = ET.fromstring("<CoSimulation></CoSimulation>")
    with pytest.raises(
        ValueError, match="CoSimulation element must have modelIdentifier attribute"
    ):
        _parse_co_simulation(cs_xml)

    # Test Unit element without name attribute
    unit_xml = ET.fromstring("<Unit></Unit>")
    with pytest.raises(ValueError, match="Unit element.*must have name attribute"):
        _parse_unit(unit_xml)

    # Test DisplayUnit element without name attribute
    display_unit_xml = ET.fromstring("<DisplayUnit></DisplayUnit>")
    with pytest.raises(
        ValueError, match="DisplayUnit element.*must have name attribute"
    ):
        _parse_display_unit(display_unit_xml)

    # Test SimpleType element without name attribute
    simple_type_xml = ET.fromstring("<SimpleType></SimpleType>")
    with pytest.raises(
        ValueError, match="SimpleType element.*must have name attribute"
    ):
        _parse_simple_type(simple_type_xml)

    # Test Item element without name attribute
    item_xml = ET.fromstring('<Item value="1"></Item>')
    with pytest.raises(ValueError, match="Item element.*must have name attribute"):
        # We need to test this inside an enumeration context
        enum_xml = ET.fromstring('<Enumeration><Item value="1"></Item></Enumeration>')
        _parse_enumeration_simple_type(enum_xml)

    # Test Item element without value attribute
    item_xml = ET.fromstring('<Item name="test"></Item>')
    with pytest.raises(ValueError, match="Item element.*must have value attribute"):
        # We need to test this inside an enumeration context
        enum_xml = ET.fromstring('<Enumeration><Item name="test"></Item></Enumeration>')
        _parse_enumeration_simple_type(enum_xml)


def test_edge_cases_and_validators():
    """Test edge cases and validation logic"""
    from mdreader.fmi2 import RealSimpleType, IntegerSimpleType, _str_to_bool

    # Test RealSimpleType with max < min (should raise error)
    with pytest.raises(ValueError, match="max.*must be >= min"):
        RealSimpleType(min_value=10.0, max_value=5.0)

    # Test IntegerSimpleType with max < min (should raise error)
    with pytest.raises(ValueError, match="max.*must be >= min"):
        IntegerSimpleType(min_value=10, max_value=5)

    # Test _str_to_bool with invalid value
    with pytest.raises(ValueError, match="Cannot convert.*to boolean"):
        _str_to_bool("invalid_boolean_value")

    # Test _str_to_bool with various valid values
    assert _str_to_bool("true") is True
    assert _str_to_bool("false") is False
    assert _str_to_bool("1") is True
    assert _str_to_bool("0") is False
    assert _str_to_bool("yes") is True
    assert _str_to_bool("no") is False
    assert _str_to_bool("on") is True
    assert _str_to_bool("off") is False
    assert _str_to_bool("") is False
    assert _str_to_bool(None) is None
    assert _str_to_bool(False) is False
    assert _str_to_bool(True) is True


def test_xml_serialization():
    """Test the to_xml() methods for all model classes"""
    from mdreader.fmi2 import (
        FmiModelDescription,
        ModelVariables,
        ScalarVariable,
        RealVariable,
        CausalityEnum,
        VariabilityEnum,
        BaseUnit,
        DisplayUnit,
        Unit,
        Item,
        RealSimpleType,
        SimpleType,
        ModelExchange,
        CoSimulation,
        SourceFiles,
        File,
        Category,
        LogCategories,
        DefaultExperiment,
        Annotation,
        Tool,
        UnknownDependency,
        VariableDependency,
        InitialUnknown,
        InitialUnknowns,
        ModelStructure,
        IntegerVariable,
        BooleanVariable,
        StringVariable,
        EnumerationVariable,
        IntegerSimpleType,
        BooleanSimpleType,
        StringSimpleType,
        EnumerationSimpleType,
        DependenciesKindEnum,
    )
    import xml.etree.ElementTree as ET

    # Test BaseUnit to_xml
    base_unit = BaseUnit(kg=1, m=1, s=-2)  # Force non-default values
    base_unit_xml = base_unit.to_xml()
    assert base_unit_xml.tag == "BaseUnit"
    assert base_unit_xml.get("kg") == "1"
    assert base_unit_xml.get("m") == "1"
    assert base_unit_xml.get("s") == "-2"

    # Test DisplayUnit to_xml
    display_unit = DisplayUnit(name="deg", factor=57.29577951308232)
    display_unit_xml = display_unit.to_xml()
    assert display_unit_xml.tag == "DisplayUnit"
    assert display_unit_xml.get("name") == "deg"
    assert display_unit_xml.get("factor") == "57.29577951308232"

    # Test Unit to_xml
    unit = Unit(name="rad", base_unit=base_unit, display_units=[display_unit])
    unit_xml = unit.to_xml()
    assert unit_xml.tag == "Unit"
    assert unit_xml.get("name") == "rad"
    assert len(unit_xml) == 2  # base_unit and display_unit

    # Test Item to_xml
    item = Item(name="Option1", value=1, description="First option")
    item_xml = item.to_xml()
    assert item_xml.tag == "Item"
    assert item_xml.get("name") == "Option1"
    assert item_xml.get("value") == "1"
    assert item_xml.get("description") == "First option"

    # Test RealSimpleType to_xml
    real_simple = RealSimpleType(
        quantity="Length", unit="m", min_value=0.0, max_value=10.0
    )
    real_simple_xml = real_simple.to_xml()
    assert real_simple_xml.tag == "Real"
    assert real_simple_xml.get("quantity") == "Length"
    assert real_simple_xml.get("unit") == "m"
    assert real_simple_xml.get("min") == "0.0"
    assert real_simple_xml.get("max") == "10.0"

    # Test IntegerSimpleType to_xml
    int_simple = IntegerSimpleType(quantity="Count", min_value=0, max_value=100)
    int_simple_xml = int_simple.to_xml()
    assert int_simple_xml.tag == "Integer"
    assert int_simple_xml.get("quantity") == "Count"
    assert int_simple_xml.get("min") == "0"
    assert int_simple_xml.get("max") == "100"

    # Test BooleanSimpleType to_xml
    bool_simple = BooleanSimpleType()
    bool_simple_xml = bool_simple.to_xml()
    assert bool_simple_xml.tag == "Boolean"

    # Test StringSimpleType to_xml
    str_simple = StringSimpleType()
    str_simple_xml = str_simple.to_xml()
    assert str_simple_xml.tag == "String"

    # Test EnumerationSimpleType to_xml
    enum_simple = EnumerationSimpleType(quantity="Options", items=[item])
    enum_simple_xml = enum_simple.to_xml()
    assert enum_simple_xml.tag == "Enumeration"
    assert enum_simple_xml.get("quantity") == "Options"
    assert len(enum_simple_xml) == 1

    # Test SimpleType to_xml with Real
    simple_type = SimpleType(
        name="LengthType", description="A length type", real=real_simple
    )
    simple_type_xml = simple_type.to_xml()
    assert simple_type_xml.tag == "SimpleType"
    assert simple_type_xml.get("name") == "LengthType"
    assert simple_type_xml.get("description") == "A length type"
    assert len(simple_type_xml) == 1
    assert simple_type_xml[0].tag == "Real"

    # Test File to_xml
    file_obj = File(name="source.c")
    file_xml = file_obj.to_xml()
    assert file_xml.tag == "File"
    assert file_xml.get("name") == "source.c"

    # Test SourceFiles to_xml
    source_files = SourceFiles(files=[file_obj])
    source_files_xml = source_files.to_xml()
    assert source_files_xml.tag == "SourceFiles"
    assert len(source_files_xml) == 1

    # Test ModelExchange to_xml
    model_exchange = ModelExchange(
        model_identifier="TestModel",
        needs_execution_tool=True,
        source_files=source_files,
    )
    model_exchange_xml = model_exchange.to_xml()
    assert model_exchange_xml.tag == "ModelExchange"
    assert model_exchange_xml.get("modelIdentifier") == "TestModel"
    assert model_exchange_xml.get("needsExecutionTool") == "true"
    assert len(model_exchange_xml) == 1

    # Test CoSimulation to_xml
    co_simulation = CoSimulation(
        model_identifier="TestModel", can_handle_variable_communication_step_size=True
    )
    co_simulation_xml = co_simulation.to_xml()
    assert co_simulation_xml.tag == "CoSimulation"
    assert co_simulation_xml.get("modelIdentifier") == "TestModel"
    assert co_simulation_xml.get("canHandleVariableCommunicationStepSize") == "true"

    # Test Category to_xml
    category = Category(name="logEvents", description="Log events")
    category_xml = category.to_xml()
    assert category_xml.tag == "Category"
    assert category_xml.get("name") == "logEvents"
    assert category_xml.get("description") == "Log events"

    # Test LogCategories to_xml
    log_categories = LogCategories(categories=[category])
    log_categories_xml = log_categories.to_xml()
    assert log_categories_xml.tag == "LogCategories"
    assert len(log_categories_xml) == 1

    # Test DefaultExperiment to_xml
    default_exp = DefaultExperiment(start_time=0.0, stop_time=10.0, step_size=0.01)
    default_exp_xml = default_exp.to_xml()
    assert default_exp_xml.tag == "DefaultExperiment"
    assert default_exp_xml.get("startTime") == "0.0"
    assert default_exp_xml.get("stopTime") == "10.0"
    assert default_exp_xml.get("stepSize") == "0.01"

    # Test Tool to_xml
    tool = Tool(name="TestTool")
    tool_xml = tool.to_xml()
    assert tool_xml.tag == "Tool"
    assert tool_xml.get("name") == "TestTool"

    # Test Annotation to_xml
    annotation = Annotation(tools=[tool])
    annotation_xml = annotation.to_xml()
    assert annotation_xml.tag == "Annotation"
    assert len(annotation_xml) == 1

    # Test UnknownDependency to_xml
    unknown_dep = UnknownDependency(
        index=1,
        dependencies=[2, 3],
        dependencies_kind=[
            DependenciesKindEnum.dependent,
            DependenciesKindEnum.constant,
        ],
    )
    unknown_dep_xml = unknown_dep.to_xml()
    assert unknown_dep_xml.tag == "Unknown"
    assert unknown_dep_xml.get("index") == "1"
    assert unknown_dep_xml.get("dependencies") == "2 3"
    assert unknown_dep_xml.get("dependenciesKind") == "dependent constant"

    # Test VariableDependency to_xml
    var_dep = VariableDependency(unknowns=[unknown_dep])
    var_dep_xml = var_dep.to_xml()
    assert var_dep_xml.tag == "VariableDependency"
    assert len(var_dep_xml) == 1

    # Test InitialUnknown to_xml
    initial_unknown = InitialUnknown(
        index=1, dependencies=[2, 3], dependencies_kind=[DependenciesKindEnum.dependent]
    )
    initial_unknown_xml = initial_unknown.to_xml()
    assert initial_unknown_xml.tag == "Unknown"  # Same tag as UnknownDependency
    assert initial_unknown_xml.get("index") == "1"

    # Test InitialUnknowns to_xml
    initial_unknowns = InitialUnknowns(unknowns=[initial_unknown])
    initial_unknowns_xml = initial_unknowns.to_xml()
    assert initial_unknowns_xml.tag == "InitialUnknowns"
    assert len(initial_unknowns_xml) == 1

    # Test ModelStructure to_xml
    model_structure = ModelStructure(outputs=var_dep, initial_unknowns=initial_unknowns)
    model_structure_xml = model_structure.to_xml()
    assert model_structure_xml.tag == "ModelStructure"
    assert len(model_structure_xml) == 2  # outputs and initial_unknowns

    # Test RealVariable to_xml
    real_var = RealVariable(
        declared_type="LengthType", unit="m", min_value=0.0, max_value=10.0, start=5.0
    )
    real_var_xml = real_var.to_xml()
    assert real_var_xml.tag == "Real"
    assert real_var_xml.get("declaredType") == "LengthType"
    assert real_var_xml.get("unit") == "m"
    assert real_var_xml.get("min") == "0.0"
    assert real_var_xml.get("max") == "10.0"
    assert real_var_xml.get("start") == "5.0"

    # Test IntegerVariable to_xml
    int_var = IntegerVariable(
        declared_type="CountType", min_value=0, max_value=100, start=50
    )
    int_var_xml = int_var.to_xml()
    assert int_var_xml.tag == "Integer"
    assert int_var_xml.get("declaredType") == "CountType"
    assert int_var_xml.get("min") == "0"
    assert int_var_xml.get("max") == "100"
    assert int_var_xml.get("start") == "50"

    # Test BooleanVariable to_xml
    bool_var = BooleanVariable(declared_type="FlagType", start=True)
    bool_var_xml = bool_var.to_xml()
    assert bool_var_xml.tag == "Boolean"
    assert bool_var_xml.get("declaredType") == "FlagType"
    assert bool_var_xml.get("start") == "true"

    # Test StringVariable to_xml
    str_var = StringVariable(declared_type="TextType", start="Hello")
    str_var_xml = str_var.to_xml()
    assert str_var_xml.tag == "String"
    assert str_var_xml.get("declaredType") == "TextType"
    assert str_var_xml.get("start") == "Hello"

    # Test EnumerationVariable to_xml
    enum_var = EnumerationVariable(
        declared_type="OptionType", min_value=1, max_value=3, start=2
    )
    enum_var_xml = enum_var.to_xml()
    assert enum_var_xml.tag == "Enumeration"
    assert enum_var_xml.get("declaredType") == "OptionType"
    assert enum_var_xml.get("min") == "1"
    assert enum_var_xml.get("max") == "3"
    assert enum_var_xml.get("start") == "2"

    # Test ScalarVariable to_xml
    scalar_var = ScalarVariable(
        name="test_var",
        value_reference=1,
        description="A test variable",
        causality=CausalityEnum.output,
        variability=VariabilityEnum.continuous,
        real=real_var,
        annotations=annotation,
    )
    scalar_var_xml = scalar_var.to_xml()
    assert scalar_var_xml.tag == "ScalarVariable"
    assert scalar_var_xml.get("name") == "test_var"
    assert scalar_var_xml.get("valueReference") == "1"
    assert scalar_var_xml.get("description") == "A test variable"
    assert scalar_var_xml.get("causality") == "output"
    assert scalar_var_xml.get("variability") == "continuous"
    assert len(scalar_var_xml) == 2  # Real element and Annotations element

    # Test ModelVariables to_xml
    model_vars = ModelVariables(variables=[scalar_var])
    model_vars_xml = model_vars.to_xml()
    assert model_vars_xml.tag == "ModelVariables"
    assert len(model_vars_xml) == 1

    # Test UnitDefinitions to_xml
    from mdreader.fmi2 import UnitDefinitions

    unit_defs = UnitDefinitions(units=[unit])
    unit_defs_xml = unit_defs.to_xml()
    assert unit_defs_xml.tag == "UnitDefinitions"
    assert len(unit_defs_xml) == 1

    # Test TypeDefinitions to_xml
    from mdreader.fmi2 import TypeDefinitions

    type_defs = TypeDefinitions(simple_types=[simple_type])
    type_defs_xml = type_defs.to_xml()
    assert type_defs_xml.tag == "TypeDefinitions"
    assert len(type_defs_xml) == 1

    # Test FmiModelDescription to_xml
    model_desc = FmiModelDescription(
        fmi_version="2.0",
        model_name="TestModel",
        guid="{12345678-1234-5678-9012-123456789012}",
        description="A test model",
        model_exchange=model_exchange,
        unit_definitions=unit_defs,
        type_definitions=type_defs,
        log_categories=log_categories,
        default_experiment=default_exp,
        vendor_annotations=annotation,
        model_variables=model_vars,
        model_structure=model_structure,
    )
    model_desc_xml = model_desc.to_xml()
    assert model_desc_xml.tag == "fmiModelDescription"
    assert model_desc_xml.get("fmiVersion") == "2.0"
    assert model_desc_xml.get("modelName") == "TestModel"
    assert model_desc_xml.get("guid") == "{12345678-1234-5678-9012-123456789012}"
    assert model_desc_xml.get("description") == "A test model"
    assert len(model_desc_xml) == 8  # All optional components

    # Test that the generated XML can be parsed back to a string
    xml_string = ET.tostring(model_desc_xml, encoding="unicode")
    assert "fmiModelDescription" in xml_string
    assert "TestModel" in xml_string
    assert "{12345678-1234-5678-9012-123456789012}" in xml_string


def test_xml_serialization_roundtrip():
    """Test that XML serialization and deserialization work correctly"""
    from mdreader.fmi2 import (
        FmiModelDescription,
        ModelVariables,
        ScalarVariable,
        RealVariable,
        CausalityEnum,
        VariabilityEnum,
    )
    import xml.etree.ElementTree as ET

    # Create a simple model
    real_var = RealVariable(unit="m", min_value=0.0, max_value=10.0, start=5.0)
    scalar_var = ScalarVariable(
        name="test_var",
        value_reference=1,
        description="A test variable",
        causality=CausalityEnum.output,
        variability=VariabilityEnum.continuous,
        real=real_var,
    )
    model_vars = ModelVariables(variables=[scalar_var])
    model_desc = FmiModelDescription(
        fmi_version="2.0",
        model_name="TestModel",
        guid="{12345678-1234-5678-9012-123456789012}",
        model_variables=model_vars,
    )

    # Convert to XML
    xml_element = model_desc.to_xml()
    xml_string = ET.tostring(xml_element, encoding="unicode")

    # Verify that the XML string contains expected elements
    assert 'fmiVersion="2.0"' in xml_string
    assert 'modelName="TestModel"' in xml_string
    assert 'guid="{12345678-1234-5678-9012-123456789012}"' in xml_string
    assert 'name="test_var"' in xml_string
    assert 'valueReference="1"' in xml_string
    assert 'causality="output"' in xml_string


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "2.0/Feedthrough.fmu",
        "2.0/BouncingBall.fmu",
        "2.0/VanDerPol.fmu",
        "2.0/Dahlquist.fmu",
        "2.0/Stair.fmu",
        "2.0/Resource.fmu",
    ],
)
def test_xml_serialization_roundtrip_with_reference_fmus(
    reference_fmu, reference_fmus_dir
):
    """Test XML serialization round-trip with reference FMUs"""
    import xml.etree.ElementTree as ET
    from mdreader.fmi2 import read_model_description, _parse_xml_to_model

    filename = (reference_fmus_dir / reference_fmu).absolute()

    # Read the original model description
    original_md = read_model_description(filename)

    # Convert to XML using to_xml() method
    xml_element = original_md.to_xml()

    # Convert XML element back to string
    xml_string = ET.tostring(xml_element, encoding="unicode")

    # Parse the XML string back to a model description
    parsed_element = ET.fromstring(xml_string)
    reconstructed_md = _parse_xml_to_model(parsed_element)

    # Compare key attributes between original and reconstructed
    assert original_md.fmi_version == reconstructed_md.fmi_version
    assert original_md.model_name == reconstructed_md.model_name
    assert original_md.guid == reconstructed_md.guid
    assert original_md.description == reconstructed_md.description
    assert original_md.author == reconstructed_md.author
    assert original_md.version == reconstructed_md.version
    assert original_md.copyright == reconstructed_md.copyright
    assert original_md.license == reconstructed_md.license
    assert original_md.generation_tool == reconstructed_md.generation_tool
    assert (
        original_md.number_of_event_indicators
        == reconstructed_md.number_of_event_indicators
    )

    # Compare variable counts
    assert len(original_md.model_variables.variables) == len(
        reconstructed_md.model_variables.variables
    )

    # Compare some variable properties
    for orig_var, recon_var in zip(
        original_md.model_variables.variables,
        reconstructed_md.model_variables.variables,
    ):
        assert orig_var.name == recon_var.name
        assert orig_var.value_reference == recon_var.value_reference
        assert orig_var.description == recon_var.description
        assert orig_var.causality == recon_var.causality
        assert orig_var.variability == recon_var.variability
        assert orig_var.initial == recon_var.initial

        # Compare variable type-specific properties
        orig_type = orig_var.get_variable_type()
        recon_type = recon_var.get_variable_type()
        assert orig_type == recon_type

        if orig_type == "Real" and orig_var.real and recon_var.real:
            assert orig_var.real.declared_type == recon_var.real.declared_type
            assert orig_var.real.unit == recon_var.real.unit
            assert orig_var.real.min_value == recon_var.real.min_value
            assert orig_var.real.max_value == recon_var.real.max_value
            assert orig_var.real.start == recon_var.real.start
        elif orig_type == "Integer" and orig_var.integer and recon_var.integer:
            assert orig_var.integer.declared_type == recon_var.integer.declared_type
            assert orig_var.integer.min_value == recon_var.integer.min_value
            assert orig_var.integer.max_value == recon_var.integer.max_value
            assert orig_var.integer.start == recon_var.integer.start
        elif orig_type == "Boolean" and orig_var.boolean and recon_var.boolean:
            assert orig_var.boolean.declared_type == recon_var.boolean.declared_type
            assert orig_var.boolean.start == recon_var.boolean.start
        elif orig_type == "String" and orig_var.string and recon_var.string:
            assert orig_var.string.declared_type == recon_var.string.declared_type
            assert orig_var.string.start == recon_var.string.start
        elif (
            orig_type == "Enumeration"
            and orig_var.enumeration
            and recon_var.enumeration
        ):
            assert (
                orig_var.enumeration.declared_type
                == recon_var.enumeration.declared_type
            )
            assert orig_var.enumeration.min_value == recon_var.enumeration.min_value
            assert orig_var.enumeration.max_value == recon_var.enumeration.max_value
            assert orig_var.enumeration.start == recon_var.enumeration.start


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "2.0/Feedthrough.fmu",
        "2.0/BouncingBall.fmu",
    ],
)
def test_xml_serialization_with_optional_components(reference_fmu, reference_fmus_dir):
    """Test XML serialization preserves optional components like ModelExchange, CoSimulation, etc."""
    import xml.etree.ElementTree as ET
    from mdreader.fmi2 import read_model_description

    filename = (reference_fmus_dir / reference_fmu).absolute()

    # Read the original model description
    original_md = read_model_description(filename)

    # Convert to XML using to_xml() method
    xml_element = original_md.to_xml()

    # Convert XML element back to string to ensure it's well-formed
    xml_string = ET.tostring(xml_element, encoding="unicode")

    # Verify that the XML string contains expected root attributes
    assert f'fmiVersion="{original_md.fmi_version}"' in xml_string
    assert f'modelName="{original_md.model_name}"' in xml_string
    assert f'guid="{original_md.guid}"' in xml_string

    # Check for optional components if they exist in the original
    if original_md.model_exchange:
        assert "<ModelExchange" in xml_string
        assert (
            f'modelIdentifier="{original_md.model_exchange.model_identifier}"'
            in xml_string
        )

    if original_md.co_simulation:
        assert "<CoSimulation" in xml_string
        assert (
            f'modelIdentifier="{original_md.co_simulation.model_identifier}"'
            in xml_string
        )

    if original_md.unit_definitions and original_md.unit_definitions.units:
        assert "<UnitDefinitions" in xml_string
        for unit in original_md.unit_definitions.units:
            assert f'name="{unit.name}"' in xml_string

    if original_md.type_definitions and original_md.type_definitions.simple_types:
        assert "<TypeDefinitions" in xml_string

    if original_md.log_categories and original_md.log_categories.categories:
        assert "<LogCategories" in xml_string

    if original_md.default_experiment:
        assert "<DefaultExperiment" in xml_string

    if original_md.model_structure:
        assert "<ModelStructure" in xml_string
        if original_md.model_structure.outputs:
            assert "<Outputs" in xml_string
        if original_md.model_structure.derivatives:
            assert "<Derivatives" in xml_string
        if original_md.model_structure.initial_unknowns:
            assert "<InitialUnknowns" in xml_string
