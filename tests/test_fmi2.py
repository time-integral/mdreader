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
