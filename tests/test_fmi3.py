import pathlib
import zipfile
from typing import Any, Dict, List

import httpx
import pytest

from mdreader.fmi3 import read_model_description, Variable

# Using a different URL for FMI 3.0 reference FMUs
REFERENCE_FMUS_URL = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.39/Reference-FMUs-0.0.39.zip"


@pytest.fixture(scope="session")
def reference_fmus_dir(tmp_path_factory) -> pathlib.Path:
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


def test_basic_parsing():
    """Basic test to ensure FMI 3.0 parsing works with a simple example."""
    # Create a minimal FMI 3.0 XML content for testing
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription
    fmiVersion="3.0"
    modelName="TestModel"
    instantiationToken="{12345678-1234-1234-1234-123456789012}"
    description="Test model for FMI 3.0">
    <ModelVariables>
        <Float64 name="testVar" valueReference="0" causality="output"/>
    </ModelVariables>
</fmiModelDescription>"""

    # Write temporary XML file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test reading from XML file
        md = read_model_description(temp_xml_path)
        assert md.fmi_version == "3.0"
        assert md.model_name == "TestModel"
        assert md.instantiation_token == "{12345678-1234-1234-1234-123456789012}"
        assert md.description == "Test model for FMI 3.0"
        assert len(md.model_variables.variables) == 1
        assert md.model_variables.variables[0].get_variable_type() == "Float64"
        var = md.model_variables.variables[0].float64
        assert var is not None
        assert var.name == "testVar"
        assert var.value_reference == 0
        assert var.causality is not None
        assert var.causality.value == "output"
    finally:
        # Clean up
        os.unlink(temp_xml_path)


@pytest.mark.parametrize(
    "reference_fmu, expected_interfaces",
    [
        (
            "3.0/Feedthrough.fmu",
            {
                "model_exchange": True,
                "co_simulation": True,
                "scheduled_execution": False,
            },
        ),
        (
            "3.0/BouncingBall.fmu",
            {
                "model_exchange": True,
                "co_simulation": True,
                "scheduled_execution": False,
            },
        ),
        (
            "3.0/VanDerPol.fmu",
            {
                "model_exchange": True,
                "co_simulation": True,
                "scheduled_execution": False,
            },
        ),
        (
            "3.0/Dahlquist.fmu",
            {
                "model_exchange": True,
                "co_simulation": True,
                "scheduled_execution": False,
            },
        ),
        (
            "3.0/Stair.fmu",
            {
                "model_exchange": True,
                "co_simulation": True,
                "scheduled_execution": False,
            },
        ),
        (
            "3.0/Resource.fmu",
            {
                "model_exchange": True,
                "co_simulation": True,
                "scheduled_execution": False,
            },
        ),
        (
            "3.0/Clocks.fmu",
            {
                "model_exchange": False,
                "co_simulation": False,
                "scheduled_execution": True,
            },
        ),
    ],
)
def test_interface_types(
    reference_fmu: str,
    expected_interfaces: Dict[str, bool],
    reference_fmus_dir: pathlib.Path,
):
    """Test that different FMI 3.0 interface types are correctly parsed."""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    # Check Model Exchange
    if expected_interfaces["model_exchange"]:
        assert md.model_exchange is not None
    else:
        assert md.model_exchange is None

    # Check Co-Simulation
    if expected_interfaces["co_simulation"]:
        assert md.co_simulation is not None
    else:
        assert md.co_simulation is None

    # Check Scheduled Execution
    if expected_interfaces["scheduled_execution"]:
        assert md.scheduled_execution is not None
    else:
        assert md.scheduled_execution is None


@pytest.mark.parametrize(
    "reference_fmu, expected_inputs, expected_outputs",
    [
        (
            "3.0/Feedthrough.fmu",
            [
                "Float32_continuous_input",
                "Float32_discrete_input",
                "Float64_continuous_input",
                "Float64_discrete_input",
                "Int8_input",
                "UInt8_input",
                "Int16_input",
                "UInt16_input",
                "Int32_input",
                "UInt32_input",
                "Int64_input",
                "UInt64_input",
                "Boolean_input",
                "String_input",
                "Binary_input",
                "Enumeration_input",
            ],
            [
                "Float32_continuous_output",
                "Float32_discrete_output",
                "Float64_continuous_output",
                "Float64_discrete_output",
                "Int8_output",
                "UInt8_output",
                "Int16_output",
                "UInt16_output",
                "Int32_output",
                "UInt32_output",
                "Int64_output",
                "UInt64_output",
                "Boolean_output",
                "String_output",
                "Binary_output",
                "Enumeration_output",
            ],
        ),
        ("3.0/BouncingBall.fmu", [], ["h", "v"]),
        ("3.0/VanDerPol.fmu", [], ["x0", "x1"]),
        ("3.0/Dahlquist.fmu", [], ["x"]),
        ("3.0/Stair.fmu", [], ["counter"]),
        ("3.0/Resource.fmu", [], ["y"]),
    ],
)
def test_scalar_variables(
    reference_fmu: str,
    expected_inputs: List[str],
    expected_outputs: List[str],
    reference_fmus_dir: pathlib.Path,
):
    """Test that scalar variables are correctly parsed from FMI 3.0 model description."""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    def get_var_name(var: Variable) -> str | None:
        """Get the name of a variable by checking which type field is set."""
        var_type = var.get_variable_type()
        if var_type:
            return getattr(var, var_type.lower()).name
        return None

    def get_var_causality(var: Variable) -> str | None:
        """Get the causality of a variable."""
        var_type = var.get_variable_type()
        if var_type:
            var_obj = getattr(var, var_type.lower())
            if var_obj.causality:
                return var_obj.causality.value
        return None

    input_vars = [
        get_var_name(var)
        for var in md.model_variables.variables
        if get_var_causality(var) == "input"
    ]

    output_vars = [
        get_var_name(var)
        for var in md.model_variables.variables
        if get_var_causality(var) == "output"
    ]

    # Filter out None values before sorting
    input_vars = [v for v in input_vars if v is not None]
    output_vars = [v for v in output_vars if v is not None]

    assert sorted(input_vars) == sorted(expected_inputs)
    assert sorted(output_vars) == sorted(expected_outputs)


@pytest.mark.parametrize(
    "reference_fmu, expected_units",
    [
        (
            "3.0/BouncingBall.fmu",
            [
                {"name": "m", "base_unit": {"m": 1}},
                {"name": "m/s", "base_unit": {"m": 1, "s": -1}},
                {"name": "m/s2", "base_unit": {"m": 1, "s": -2}},
            ],
        ),
        # Feedthrough and others may not have unit definitions
        ("3.0/Feedthrough.fmu", []),
        ("3.0/VanDerPol.fmu", []),
        ("3.0/Dahlquist.fmu", []),
        ("3.0/Stair.fmu", []),
        ("3.0/Resource.fmu", []),
    ],
)
def test_unit_definitions(
    reference_fmu: str,
    expected_units: List[Dict[str, Any]],
    reference_fmus_dir: pathlib.Path,
):
    """Test that unit definitions are properly parsed with correct values"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if expected_units:
        assert md.unit_definitions is not None
        assert len(md.unit_definitions.units) == len(expected_units)

        for i, expected_unit in enumerate(expected_units):
            unit = md.unit_definitions.units[i]
            assert unit.name == expected_unit["name"]

            if "base_unit" in expected_unit and unit.base_unit:
                expected_base = expected_unit["base_unit"]
                for attr, expected_val in expected_base.items():
                    assert getattr(unit.base_unit, attr) == expected_val
    else:
        # For FMUs without unit definitions, ensure it's None or empty
        if md.unit_definitions:
            assert len(md.unit_definitions.units) == 0


@pytest.mark.parametrize(
    "reference_fmu, expected_types",
    [
        (
            "3.0/Feedthrough.fmu",
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
            "3.0/BouncingBall.fmu",
            [
                {
                    "name": "Position",
                    "type_category": "Float64",
                    "quantity": "Position",
                    "unit": "m",
                },
                {
                    "name": "Velocity",
                    "type_category": "Float64",
                    "quantity": "Velocity",
                    "unit": "m/s",
                },
                {
                    "name": "Acceleration",
                    "type_category": "Float64",
                    "quantity": "Acceleration",
                    "unit": "m/s2",
                },
            ],
        ),
    ],
)
def test_type_definitions(
    reference_fmu: str,
    expected_types: List[Dict[str, Any]],
    reference_fmus_dir: pathlib.Path,
):
    """Test that type definitions are properly parsed with correct values"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if expected_types:
        assert md.type_definitions is not None
        assert len(md.type_definitions.types) == len(expected_types)

        for i, expected_type in enumerate(expected_types):
            type_def = md.type_definitions.types[i]
            assert type_def.name == expected_type["name"]
            assert type_def.get_type_category() == expected_type["type_category"]

            if "items" in expected_type and type_def.enumeration:
                for j, expected_item in enumerate(expected_type["items"]):
                    item = type_def.enumeration.items[j]
                    assert item.name == expected_item["name"]
                    assert item.value == expected_item["value"]
                    assert item.description == expected_item["description"]
            elif "quantity" in expected_type:
                if type_def.float64:
                    assert type_def.float64.quantity == expected_type["quantity"]
                if "unit" in expected_type:
                    if type_def.float64:
                        assert type_def.float64.unit == expected_type["unit"]
    else:
        # For FMUs without type definitions, ensure it's None or empty
        if md.type_definitions:
            assert len(md.type_definitions.types) == 0


@pytest.mark.parametrize(
    "reference_fmu, expected_variables",
    [
        # Test specific variables from Feedthrough
        (
            "3.0/Feedthrough.fmu",
            [
                {
                    "name": "time",
                    "value_reference": 0,
                    "causality": "independent",
                    "variability": "continuous",
                    "type": "Float64",
                },
                {
                    "name": "Float64_continuous_input",
                    "value_reference": 7,
                    "causality": "input",
                    "variability": "continuous",
                    "type": "Float64",
                    "start": [0.0],
                },
                {
                    "name": "Int32_input",
                    "value_reference": 19,
                    "causality": "input",
                    "variability": "continuous",
                    "type": "Int32",
                    "start": [0],
                },
                {
                    "name": "Boolean_input",
                    "value_reference": 27,
                    "causality": "input",
                    "variability": "continuous",
                    "type": "Boolean",
                    "start": [False],
                },
                {
                    "name": "Enumeration_input",
                    "value_reference": 33,
                    "causality": "input",
                    "variability": "continuous",
                    "type": "Enumeration",
                    "declared_type": "Option",
                    "start": [1],
                },
            ],
        ),
        # Test specific variables from BouncingBall
        (
            "3.0/BouncingBall.fmu",
            [
                {
                    "name": "time",
                    "value_reference": 0,
                    "causality": "independent",
                    "variability": "continuous",
                    "description": "Simulation time",
                    "type": "Float64",
                },
                {
                    "name": "h",
                    "value_reference": 1,
                    "causality": "output",
                    "variability": "continuous",
                    "initial": "exact",
                    "description": "Position of the ball",
                    "type": "Float64",
                    "start": [1.0],
                    "declared_type": "Position",
                },
                {
                    "name": "g",
                    "value_reference": 5,
                    "causality": "parameter",
                    "variability": "fixed",
                    "initial": "exact",
                    "description": "Gravity acting on the ball",
                    "type": "Float64",
                    "start": [-9.81],
                    "declared_type": "Acceleration",
                },
            ],
        ),
    ],
)
def test_variable_properties(
    reference_fmu: str,
    expected_variables: List[Dict[str, Any]],
    reference_fmus_dir: pathlib.Path,
):
    """Test that variable properties are correctly parsed from FMI 3.0 XML"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    # Create a mapping of variable names to variables for easy lookup
    var_map = {}
    for var in md.model_variables.variables:
        var_type = var.get_variable_type()
        if var_type:
            var_map[var.__getattribute__(var_type.lower()).name] = var

    for expected_var in expected_variables:
        var_name = expected_var["name"]
        assert var_name in var_map, f"Variable {var_name} not found in model"

        var = var_map[var_name]
        var_type = var.get_variable_type()

        # Check common properties
        assert var.__getattribute__(var_type.lower()).name == expected_var["name"]
        assert (
            var.__getattribute__(var_type.lower()).value_reference
            == expected_var["value_reference"]
        )
        assert (
            var.__getattribute__(var_type.lower()).causality.value
            == expected_var["causality"]
        )
        assert (
            var.__getattribute__(var_type.lower()).variability.value
            == expected_var["variability"]
        )

        if "description" in expected_var:
            assert (
                var.__getattribute__(var_type.lower()).description
                == expected_var["description"]
            )

        if "initial" in expected_var:
            assert (
                var.__getattribute__(var_type.lower()).initial.value
                == expected_var["initial"]
            )

        # Check variable-specific properties
        assert var_type == expected_var["type"]

        if "start" in expected_var:
            assert var.__getattribute__(var_type.lower()).start == expected_var["start"]
        if "declared_type" in expected_var:
            assert (
                var.__getattribute__(var_type.lower()).declared_type
                == expected_var["declared_type"]
            )


@pytest.mark.parametrize(
    "reference_fmu, expected_model_exchange",
    [
        (
            "3.0/Feedthrough.fmu",
            {
                "model_identifier": "Feedthrough",
                "can_get_and_set_fmu_state": True,
            },
        ),
        (
            "3.0/BouncingBall.fmu",
            {
                "model_identifier": "BouncingBall",
                "can_get_and_set_fmu_state": True,
            },
        ),
        (
            "3.0/VanDerPol.fmu",
            {
                "model_identifier": "VanDerPol",
                "can_get_and_set_fmu_state": True,
                "provides_directional_derivatives": True,
                "provides_adjoint_derivatives": True,
            },
        ),
        (
            "3.0/Dahlquist.fmu",
            {
                "model_identifier": "Dahlquist",
                "can_get_and_set_fmu_state": True,
            },
        ),
        (
            "3.0/Stair.fmu",
            {
                "model_identifier": "Stair",
                "can_get_and_set_fmu_state": True,
            },
        ),
        (
            "3.0/Resource.fmu",
            {
                "model_identifier": "Resource",
                "can_get_and_set_fmu_state": True,
            },
        ),
    ],
)
def test_model_exchange_interface(
    reference_fmu: str,
    expected_model_exchange: Dict[str, Any],
    reference_fmus_dir: pathlib.Path,
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
        md.model_exchange.can_get_and_set_fmu_state
        == expected_model_exchange["can_get_and_set_fmu_state"]
    )
    # Check optional directional/adjoint derivatives if expected
    if "provides_directional_derivatives" in expected_model_exchange:
        assert (
            md.model_exchange.provides_directional_derivatives
            == expected_model_exchange["provides_directional_derivatives"]
        )
    if "provides_adjoint_derivatives" in expected_model_exchange:
        assert (
            md.model_exchange.provides_adjoint_derivatives
            == expected_model_exchange["provides_adjoint_derivatives"]
        )


@pytest.mark.parametrize(
    "reference_fmu, expected_co_simulation",
    [
        (
            "3.0/Feedthrough.fmu",
            {
                "model_identifier": "Feedthrough",
                "can_handle_variable_communication_step_size": True,
            },
        ),
        (
            "3.0/BouncingBall.fmu",
            {
                "model_identifier": "BouncingBall",
                "can_handle_variable_communication_step_size": True,
            },
        ),
        (
            "3.0/VanDerPol.fmu",
            {
                "model_identifier": "VanDerPol",
                "can_handle_variable_communication_step_size": True,
            },
        ),
        (
            "3.0/Dahlquist.fmu",
            {
                "model_identifier": "Dahlquist",
                "can_handle_variable_communication_step_size": True,
            },
        ),
        (
            "3.0/Stair.fmu",
            {
                "model_identifier": "Stair",
                "can_handle_variable_communication_step_size": True,
            },
        ),
        (
            "3.0/Resource.fmu",
            {
                "model_identifier": "Resource",
                "can_handle_variable_communication_step_size": True,
            },
        ),
    ],
)
def test_co_simulation_interface(
    reference_fmu: str,
    expected_co_simulation: Dict[str, Any],
    reference_fmus_dir: pathlib.Path,
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


@pytest.mark.parametrize(
    "reference_fmu, expected_default_experiment",
    [
        (
            "3.0/Feedthrough.fmu",
            {"stop_time": 2.0},
        ),
        (
            "3.0/BouncingBall.fmu",
            {"start_time": 0.0, "stop_time": 3.0, "step_size": 1e-2},
        ),
        (
            "3.0/VanDerPol.fmu",
            {"start_time": 0.0, "stop_time": 20.0, "step_size": 1e-2},
        ),
        (
            "3.0/Dahlquist.fmu",
            {"start_time": 0.0, "stop_time": 10.0, "step_size": 0.1},
        ),
        (
            "3.0/Stair.fmu",
            {"start_time": 0.0, "stop_time": 10.0, "step_size": 0.2},
        ),
        (
            "3.0/Resource.fmu",
            {"start_time": 0.0, "stop_time": 1.0},
        ),
        (
            "3.0/Clocks.fmu",
            {"stop_time": 10.0, "step_size": 1.0},
        ),
    ],
)
def test_default_experiment_values(
    reference_fmu: str,
    expected_default_experiment: Dict[str, Any],
    reference_fmus_dir: pathlib.Path,
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
            "3.0/Feedthrough.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
        (
            "3.0/BouncingBall.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
        (
            "3.0/VanDerPol.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
        (
            "3.0/Dahlquist.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
        (
            "3.0/Stair.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
        (
            "3.0/Clocks.fmu",
            [
                {"name": "logEvents", "description": "Log events"},
                {"name": "logStatusError", "description": "Log error messages"},
            ],
        ),
    ],
)
def test_log_categories(
    reference_fmu: str,
    expected_log_categories: List[Dict[str, str]],
    reference_fmus_dir: pathlib.Path,
):
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
        "3.0/BouncingBall.fmu",
        "3.0/VanDerPol.fmu",
        "3.0/Dahlquist.fmu",
        "3.0/Stair.fmu",
        "3.0/Feedthrough.fmu",
    ],
)
def test_model_structure_outputs(reference_fmu: str, reference_fmus_dir: pathlib.Path):
    """Test that ModelStructure outputs are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if md.model_structure and md.model_structure.outputs:
        # Check that outputs exist and have correct structure
        for unknown in md.model_structure.outputs:
            assert isinstance(unknown.value_reference, int)
            if unknown.dependencies:
                assert all(isinstance(dep, int) for dep in unknown.dependencies)
            if unknown.dependencies_kind:
                assert all(isinstance(dep, str) for dep in unknown.dependencies_kind)


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "3.0/BouncingBall.fmu",
        "3.0/VanDerPol.fmu",
        "3.0/Dahlquist.fmu",
    ],
)
def test_model_structure_derivatives(
    reference_fmu: str, reference_fmus_dir: pathlib.Path
):
    """Test that ModelStructure derivatives are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if md.model_structure and md.model_structure.continuous_state_derivatives:
        # Check that derivatives exist and have correct structure
        for unknown in md.model_structure.continuous_state_derivatives:
            assert isinstance(unknown.value_reference, int)
            if unknown.dependencies:
                assert all(isinstance(dep, int) for dep in unknown.dependencies)
            if unknown.dependencies_kind:
                assert all(isinstance(dep, str) for dep in unknown.dependencies_kind)


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "3.0/BouncingBall.fmu",
        "3.0/VanDerPol.fmu",
        "3.0/Dahlquist.fmu",
        "3.0/Feedthrough.fmu",
        "3.0/Resource.fmu",
    ],
)
def test_model_structure_initial_unknowns(
    reference_fmu: str, reference_fmus_dir: pathlib.Path
):
    """Test that ModelStructure initial unknowns are correctly parsed"""
    filename = (reference_fmus_dir / reference_fmu).absolute()
    md = read_model_description(filename)

    if md.model_structure and md.model_structure.initial_unknowns:
        # Check that initial unknowns exist and have correct structure
        for unknown in md.model_structure.initial_unknowns:
            assert isinstance(unknown.value_reference, int)
            if unknown.dependencies:
                assert all(isinstance(dep, int) for dep in unknown.dependencies)
            if unknown.dependencies_kind:
                assert all(isinstance(dep, str) for dep in unknown.dependencies_kind)


def test_variable_validation():
    """Test validation logic in Float32Type and Float64Type"""
    from pydantic import ValidationError
    from mdreader.fmi3 import Float32Type, Float64Type

    # Test Float32Type validation - max >= min
    with pytest.raises(ValidationError, match=r"Float32Type: max .* must be >= min"):
        Float32Type(min_value=2, max_value=1)

    # Test Float64Type validation - max >= min
    with pytest.raises(ValidationError, match=r"Float64Type: max .* must be >= min"):
        Float64Type(min_value=2, max_value=1)

    # Valid cases should not raise
    Float32Type(min_value=1, max_value=2)
    Float64Type(min_value=1, max_value=2)


def test_read_model_description_with_xml_file():
    """Test reading model description from XML file directly"""
    # Create a minimal FMI 3.0 XML content for testing
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription
    fmiVersion="3.0"
    modelName="TestModel"
    instantiationToken="{12345678-1234-1234-1234-123456789012}"
    description="Test model for FMI 3.0">
    <ModelVariables>
        <Float64 name="testVar" valueReference="0" causality="output"/>
    </ModelVariables>
</fmiModelDescription>"""

    # Write temporary XML file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test reading from XML file
        md = read_model_description(temp_xml_path)
        assert md.fmi_version == "3.0"
        assert md.model_name == "TestModel"
        assert md.instantiation_token == "{12345678-1234-1234-1234-123456789012}"
        assert len(md.model_variables.variables) == 1
        assert md.model_variables.variables[0].get_variable_type() == "Float64"
    finally:
        # Clean up
        import os

        os.unlink(temp_xml_path)


def test_read_model_description_with_directory():
    """Test reading model description from directory containing modelDescription.xml"""
    import tempfile
    import os

    # Create a temporary directory with modelDescription.xml
    with tempfile.TemporaryDirectory() as temp_dir:
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription
    fmiVersion="3.0"
    modelName="TestModelDir"
    instantiationToken="{12345678-1234-1234-1234-123456789012}"
    description="Test model from directory">
    <ModelVariables>
        <Float64 name="testVar" valueReference="0" causality="output"/>
    </ModelVariables>
</fmiModelDescription>"""

        xml_path = os.path.join(temp_dir, "modelDescription.xml")
        with open(xml_path, "w") as f:
            f.write(xml_content)

        # Test reading from directory
        md = read_model_description(temp_dir)
        assert md.fmi_version == "3.0"
        assert md.model_name == "TestModelDir"
        assert md.instantiation_token == "{12345678-1234-1234-1234-123456789012}"
        assert len(md.model_variables.variables) == 1
        assert md.model_variables.variables[0].get_variable_type() == "Float64"


def test_enum_values():
    """Test that enum values are correctly defined and accessible"""
    from mdreader.fmi3 import CausalityEnum, VariabilityEnum, InitialEnum

    # Test CausalityEnum values
    assert CausalityEnum.parameter.value == "parameter"
    assert CausalityEnum.input.value == "input"
    assert CausalityEnum.output.value == "output"
    assert CausalityEnum.local.value == "local"
    assert CausalityEnum.independent.value == "independent"
    assert CausalityEnum.structuralParameter.value == "structuralParameter"

    # Test VariabilityEnum values
    assert VariabilityEnum.constant.value == "constant"
    assert VariabilityEnum.fixed.value == "fixed"
    assert VariabilityEnum.tunable.value == "tunable"
    assert VariabilityEnum.discrete.value == "discrete"
    assert VariabilityEnum.continuous.value == "continuous"

    # Test InitialEnum values
    assert InitialEnum.exact.value == "exact"
    assert InitialEnum.approx.value == "approx"
    assert InitialEnum.calculated.value == "calculated"


def test_scheduled_execution_interface(reference_fmus_dir: pathlib.Path):
    """Test that Scheduled Execution interface is correctly parsed from Clocks.fmu"""
    filename = (reference_fmus_dir / "3.0/Clocks.fmu").absolute()
    md = read_model_description(filename)

    # Clocks.fmu should have ScheduledExecution but not ModelExchange or CoSimulation
    assert md.scheduled_execution is not None
    assert md.scheduled_execution.model_identifier == "Clocks"
    assert md.model_exchange is None
    assert md.co_simulation is None


def test_clock_variables(reference_fmus_dir: pathlib.Path):
    """Test that Clock variables are correctly parsed from Clocks.fmu"""
    filename = (reference_fmus_dir / "3.0/Clocks.fmu").absolute()
    md = read_model_description(filename)

    # Check that we have clock variables
    clock_vars = [
        var
        for var in md.model_variables.variables
        if var.get_variable_type() == "Clock"
    ]

    # Clocks.fmu has 4 clock variables: inClock1, inClock2, inClock3, outClock
    assert len(clock_vars) == 4

    # Check clock names
    clock_names = []
    for var in clock_vars:
        if var.clock is not None:
            clock_names.append(var.clock.name)
    assert "inClock1" in clock_names
    assert "inClock2" in clock_names
    assert "inClock3" in clock_names
    assert "outClock" in clock_names


def test_structural_parameters(reference_fmus_dir: pathlib.Path):
    """Test that structural parameters are correctly parsed from StateSpace.fmu"""
    filename = (reference_fmus_dir / "3.0/StateSpace.fmu").absolute()
    md = read_model_description(filename)

    # Find structural parameters
    def get_var_causality(var: Variable):
        var_type = var.get_variable_type()
        if var_type:
            var_obj = getattr(var, var_type.lower())
            if var_obj.causality:
                return var_obj.causality.value
        return None

    structural_params = [
        var
        for var in md.model_variables.variables
        if get_var_causality(var) == "structuralParameter"
    ]

    # StateSpace.fmu has 3 structural parameters: m, n, r
    assert len(structural_params) == 3


def test_variable_with_declared_type(reference_fmus_dir: pathlib.Path):
    """Test that declaredType attribute is correctly parsed"""
    filename = (reference_fmus_dir / "3.0/BouncingBall.fmu").absolute()
    md = read_model_description(filename)

    # Find h which should have declaredType="Position"
    h_var = None
    for var in md.model_variables.variables:
        if (
            var.get_variable_type() == "Float64"
            and var.float64 is not None
            and var.float64.name == "h"
        ):
            h_var = var.float64
            break

    assert h_var is not None
    assert h_var.declared_type == "Position"
    assert h_var.causality is not None
    assert h_var.causality.value == "output"
    assert h_var.variability is not None
    assert h_var.variability.value == "continuous"


def test_discrete_variable(reference_fmus_dir: pathlib.Path):
    """Test that discrete variables are correctly parsed from Stair.fmu"""
    filename = (reference_fmus_dir / "3.0/Stair.fmu").absolute()
    md = read_model_description(filename)

    # Find the counter variable which is Int32 discrete output
    counter = None
    for var in md.model_variables.variables:
        if (
            var.get_variable_type() == "Int32"
            and var.int32 is not None
            and var.int32.name == "counter"
        ):
            counter = var.int32
            break

    assert counter is not None
    assert counter.causality is not None
    assert counter.causality.value == "output"
    assert counter.variability is not None
    assert counter.variability.value == "discrete"
    assert counter.start == [1]
    assert counter.max_value == 10
