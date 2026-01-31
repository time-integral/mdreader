"""FMI 3.0 model description parser and data models"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
import zipfile
from enum import Enum
from typing import Annotated, Optional
from xml.etree.ElementTree import Element

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "read_model_description",
    "CausalityEnum",
    "VariabilityEnum",
    "InitialEnum",
    "VariableNamingConventionEnum",
    "BaseUnit",
    "DisplayUnit",
    "Unit",
    "Item",
    "Float32Type",
    "Float64Type",
    "Int8Type",
    "Int16Type",
    "Int32Type",
    "Int64Type",
    "UInt8Type",
    "UInt16Type",
    "UInt32Type",
    "UInt64Type",
    "BooleanType",
    "StringType",
    "BinaryType",
    "EnumerationType",
    "ClockType",
    "Type",
    "File",
    "SourceFiles",
    "ModelExchange",
    "CoSimulation",
    "ScheduledExecution",
    "Category",
    "LogCategories",
    "DefaultExperiment",
    "Tool",
    "Annotation",
    "Unknown",
    "ModelStructure",
    "Float32Variable",
    "Float64Variable",
    "Int8Variable",
    "Int16Variable",
    "Int32Variable",
    "Int64Variable",
    "UInt8Variable",
    "UInt16Variable",
    "UInt32Variable",
    "UInt64Variable",
    "BooleanVariable",
    "StringVariable",
    "BinaryVariable",
    "EnumerationVariable",
    "ClockVariable",
    "Variable",
    "UnitDefinitions",
    "TypeDefinitions",
    "FmiModelDescription",
    "Dimension",
]


def read_model_description(filename: str | pathlib.Path) -> "FmiModelDescription":
    """Read and parse an FMI 3.0 model description XML file

    Args:
        filename (str | pathlib.Path): Path to the FMI 3.0 model description XML file or FMU directory or FMU file

    Returns:
        FmiModelDescription: Parsed FMI 3.0 model description
    """
    filename = pathlib.Path(filename)
    if filename.suffix == ".xml":
        filename = filename
    elif filename.is_dir():
        filename = filename / "modelDescription.xml"
    elif filename.suffix == ".fmu":
        with zipfile.ZipFile(filename, "r") as zf:
            with zf.open("modelDescription.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                return _parse_xml_to_model(root)
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Must be .xml, .fmu, or directory"
        )

    tree = ET.parse(filename)
    root = tree.getroot()
    return _parse_xml_to_model(root)


# Enums for FMI 3.0
class CausalityEnum(str, Enum):
    parameter = "parameter"
    calculatedParameter = "calculatedParameter"
    input = "input"
    output = "output"
    local = "local"
    independent = "independent"
    structuralParameter = "structuralParameter"


class VariabilityEnum(str, Enum):
    constant = "constant"
    fixed = "fixed"
    tunable = "tunable"
    discrete = "discrete"
    continuous = "continuous"


class InitialEnum(str, Enum):
    exact = "exact"
    approx = "approx"
    calculated = "calculated"


class VariableNamingConventionEnum(str, Enum):
    flat = "flat"
    structured = "structured"


class BaseUnit(BaseModel):
    """Base unit definition with SI base units"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    kg: Annotated[
        int | None,
        Field(default=0, alias="kg", description='Exponent of SI base unit "kg"'),
    ] = 0
    m: Annotated[
        int | None,
        Field(default=0, alias="m", description='Exponent of SI base unit "m"'),
    ] = 0
    s: Annotated[
        int | None,
        Field(default=0, alias="s", description='Exponent of SI base unit "s"'),
    ] = 0
    a: Annotated[
        int | None,
        Field(default=0, alias="A", description='Exponent of SI base unit "A"'),
    ] = 0
    k: Annotated[
        int | None,
        Field(default=0, alias="K", description='Exponent of SI base unit "K"'),
    ] = 0
    mol: Annotated[
        int | None,
        Field(default=0, alias="mol", description='Exponent of SI base unit "mol"'),
    ] = 0
    cd: Annotated[
        int | None,
        Field(default=0, alias="cd", description='Exponent of SI base unit "cd"'),
    ] = 0
    rad: Annotated[
        int | None,
        Field(default=0, alias="rad", description='Exponent of SI derived unit "rad"'),
    ] = 0
    factor: Annotated[
        float | None,
        Field(default=1.0, alias="factor", description="Factor for unit conversion"),
    ] = 1.0
    offset: Annotated[
        float | None,
        Field(default=0.0, alias="offset", description="Offset for unit conversion"),
    ] = 0.0

    def to_xml(self) -> Element:
        """Convert BaseUnit to XML Element"""
        element = Element("BaseUnit")
        if self.kg is not None and self.kg != 0:
            element.set("kg", str(self.kg))
        if self.m is not None and self.m != 0:
            element.set("m", str(self.m))
        if self.s is not None and self.s != 0:
            element.set("s", str(self.s))
        if self.a is not None and self.a != 0:
            element.set("A", str(self.a))
        if self.k is not None and self.k != 0:
            element.set("K", str(self.k))
        if self.mol is not None and self.mol != 0:
            element.set("mol", str(self.mol))
        if self.cd is not None and self.cd != 0:
            element.set("cd", str(self.cd))
        if self.rad is not None and self.rad != 0:
            element.set("rad", str(self.rad))
        if self.factor is not None and self.factor != 1.0:
            element.set("factor", str(self.factor))
        if self.offset is not None and self.offset != 0.0:
            element.set("offset", str(self.offset))
        return element


class DisplayUnit(BaseModel):
    """Display unit definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Name of DisplayUnit element, e.g. <Unit name="rad"/>, <DisplayUnit name="deg" factor="57.29..."/>. "name" must be unique with respect to all other "names" of the DisplayUnit definitions of the same Unit (different Unit elements may have the same DisplayUnit names).',
        ),
    ]
    factor: Annotated[
        float | None,
        Field(
            default=1.0,
            alias="factor",
            description="Factor for display unit conversion",
        ),
    ] = 1.0
    offset: Annotated[
        float | None,
        Field(
            default=0.0,
            alias="offset",
            description="Offset for display unit conversion",
        ),
    ] = 0.0

    def to_xml(self) -> Element:
        """Convert DisplayUnit to XML Element"""
        element = Element("DisplayUnit")
        element.set("name", self.name)
        if self.factor is not None and self.factor != 1.0:
            element.set("factor", str(self.factor))
        if self.offset is not None and self.offset != 0.0:
            element.set("offset", str(self.offset))
        return element


class Unit(BaseModel):
    """Unit definition with base unit and display units"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Name of Unit element, e.g. "N.m", "Nm",  "%/s". "name" must be unique will respect to all other elements of the UnitDefinitions list. The variable values of fmi2SetXXX and fmi2GetXXX are with respect to this unit.',
        ),
    ]
    base_unit: Annotated[
        BaseUnit | None,
        Field(
            default=None,
            alias="BaseUnit",
            description="BaseUnit_value = factor*Unit_value + offset",
        ),
    ] = None
    display_units: Annotated[
        list[DisplayUnit] | None,
        Field(
            default=None,
            alias="DisplayUnit",
            description="DisplayUnit_value = factor*Unit_value + offset",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Unit to XML Element"""
        element = Element("Unit")
        element.set("name", self.name)
        if self.base_unit is not None:
            element.append(self.base_unit.to_xml())
        if self.display_units is not None:
            for display_unit in self.display_units:
                element.append(display_unit.to_xml())
        return element


class Item(BaseModel):
    """Enumeration item"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str, Field(..., alias="name", description="Name of the enumeration item")
    ]
    value: Annotated[
        int,
        Field(
            ...,
            alias="value",
            description="Value of the enumeration item. Must be a unique number in the same enumeration",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            alias="description",
            description="Description of the enumeration item",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Item to XML Element"""
        element = Element("Item")
        element.set("name", self.name)
        element.set("value", str(self.value))
        if self.description is not None:
            element.set("description", self.description)
        return element


class File(BaseModel):
    """Source file definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[str, Field(..., alias="name")]

    def to_xml(self) -> Element:
        """Convert File to XML Element"""
        element = Element("File")
        element.set("name", self.name)
        return element


class SourceFiles(BaseModel):
    """List of source files"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    files: Annotated[list[File], Field(..., alias="File")]

    def to_xml(self) -> Element:
        """Convert SourceFiles to XML Element"""
        element = Element("SourceFiles")
        if self.files is not None:
            for file in self.files:
                element.append(file.to_xml())
        return element


class ModelExchange(BaseModel):
    """Model Exchange interface definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    model_identifier: Annotated[str, Field(..., alias="modelIdentifier")]
    needs_execution_tool: Annotated[
        bool | None, Field(default=False, alias="needsExecutionTool")
    ] = False
    can_be_instantiated_only_once_per_process: Annotated[
        bool | None,
        Field(default=False, alias="canBeInstantiatedOnlyOncePerProcess"),
    ] = False
    can_get_and_set_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canGetAndSetFMUState")
    ] = False
    can_serialize_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canSerializeFMUState")
    ] = False
    provides_directional_derivatives: Annotated[
        bool | None, Field(default=False, alias="providesDirectionalDerivatives")
    ] = False
    provides_adjoint_derivatives: Annotated[
        bool | None, Field(default=False, alias="providesAdjointDerivatives")
    ] = False
    provides_per_element_dependencies: Annotated[
        bool | None, Field(default=False, alias="providesPerElementDependencies")
    ] = False
    needs_completed_integrator_step: Annotated[
        bool | None, Field(default=False, alias="needsCompletedIntegratorStep")
    ] = False
    provides_evaluate_discrete_states: Annotated[
        bool | None, Field(default=False, alias="providesEvaluateDiscreteStates")
    ] = False

    def to_xml(self) -> Element:
        """Convert ModelExchange to XML Element"""
        element = Element("ModelExchange")
        element.set("modelIdentifier", self.model_identifier)
        if self.needs_execution_tool is not None and self.needs_execution_tool:
            element.set("needsExecutionTool", str(self.needs_execution_tool).lower())
        if (
            self.can_be_instantiated_only_once_per_process is not None
            and self.can_be_instantiated_only_once_per_process
        ):
            element.set(
                "canBeInstantiatedOnlyOncePerProcess",
                str(self.can_be_instantiated_only_once_per_process).lower(),
            )
        if (
            self.can_get_and_set_fmu_state is not None
            and self.can_get_and_set_fmu_state
        ):
            element.set(
                "canGetAndSetFMUState", str(self.can_get_and_set_fmu_state).lower()
            )
        if self.can_serialize_fmu_state is not None and self.can_serialize_fmu_state:
            element.set(
                "canSerializeFMUState", str(self.can_serialize_fmu_state).lower()
            )
        if (
            self.provides_directional_derivatives is not None
            and self.provides_directional_derivatives
        ):
            element.set(
                "providesDirectionalDerivatives",
                str(self.provides_directional_derivatives).lower(),
            )
        if (
            self.provides_adjoint_derivatives is not None
            and self.provides_adjoint_derivatives
        ):
            element.set(
                "providesAdjointDerivatives",
                str(self.provides_adjoint_derivatives).lower(),
            )
        if (
            self.provides_per_element_dependencies is not None
            and self.provides_per_element_dependencies
        ):
            element.set(
                "providesPerElementDependencies",
                str(self.provides_per_element_dependencies).lower(),
            )
        if (
            self.needs_completed_integrator_step is not None
            and self.needs_completed_integrator_step
        ):
            element.set(
                "needsCompletedIntegratorStep",
                str(self.needs_completed_integrator_step).lower(),
            )
        if (
            self.provides_evaluate_discrete_states is not None
            and self.provides_evaluate_discrete_states
        ):
            element.set(
                "providesEvaluateDiscreteStates",
                str(self.provides_evaluate_discrete_states).lower(),
            )
        return element


class CoSimulation(BaseModel):
    """Co-Simulation interface definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    model_identifier: Annotated[str, Field(..., alias="modelIdentifier")]
    needs_execution_tool: Annotated[
        bool | None, Field(default=False, alias="needsExecutionTool")
    ] = False
    can_be_instantiated_only_once_per_process: Annotated[
        bool | None,
        Field(default=False, alias="canBeInstantiatedOnlyOncePerProcess"),
    ] = False
    can_get_and_set_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canGetAndSetFMUState")
    ] = False
    can_serialize_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canSerializeFMUState")
    ] = False
    provides_directional_derivatives: Annotated[
        bool | None, Field(default=False, alias="providesDirectionalDerivatives")
    ] = False
    provides_adjoint_derivatives: Annotated[
        bool | None, Field(default=False, alias="providesAdjointDerivatives")
    ] = False
    provides_per_element_dependencies: Annotated[
        bool | None, Field(default=False, alias="providesPerElementDependencies")
    ] = False
    can_handle_variable_communication_step_size: Annotated[
        bool | None,
        Field(default=False, alias="canHandleVariableCommunicationStepSize"),
    ] = False
    fixed_internal_step_size: Annotated[
        float | None, Field(default=None, alias="fixedInternalStepSize")
    ] = None
    max_output_derivative_order: Annotated[
        int | None, Field(default=0, alias="maxOutputDerivativeOrder")
    ] = 0
    recommended_intermediate_input_smoothness: Annotated[
        int | None, Field(default=0, alias="recommendedIntermediateInputSmoothness")
    ] = 0
    provides_intermediate_update: Annotated[
        bool | None, Field(default=False, alias="providesIntermediateUpdate")
    ] = False
    might_return_early_from_do_step: Annotated[
        bool | None, Field(default=False, alias="mightReturnEarlyFromDoStep")
    ] = False
    can_return_early_after_intermediate_update: Annotated[
        bool | None,
        Field(default=False, alias="canReturnEarlyAfterIntermediateUpdate"),
    ] = False
    has_event_mode: Annotated[
        bool | None, Field(default=False, alias="hasEventMode")
    ] = False
    provides_evaluate_discrete_states: Annotated[
        bool | None, Field(default=False, alias="providesEvaluateDiscreteStates")
    ] = False

    def to_xml(self) -> Element:
        """Convert CoSimulation to XML Element"""
        element = Element("CoSimulation")
        element.set("modelIdentifier", self.model_identifier)
        if self.needs_execution_tool is not None and self.needs_execution_tool:
            element.set("needsExecutionTool", str(self.needs_execution_tool).lower())
        if (
            self.can_be_instantiated_only_once_per_process is not None
            and self.can_be_instantiated_only_once_per_process
        ):
            element.set(
                "canBeInstantiatedOnlyOncePerProcess",
                str(self.can_be_instantiated_only_once_per_process).lower(),
            )
        if (
            self.can_get_and_set_fmu_state is not None
            and self.can_get_and_set_fmu_state
        ):
            element.set(
                "canGetAndSetFMUState", str(self.can_get_and_set_fmu_state).lower()
            )
        if self.can_serialize_fmu_state is not None and self.can_serialize_fmu_state:
            element.set(
                "canSerializeFMUState", str(self.can_serialize_fmu_state).lower()
            )
        if (
            self.provides_directional_derivatives is not None
            and self.provides_directional_derivatives
        ):
            element.set(
                "providesDirectionalDerivatives",
                str(self.provides_directional_derivatives).lower(),
            )
        if (
            self.provides_adjoint_derivatives is not None
            and self.provides_adjoint_derivatives
        ):
            element.set(
                "providesAdjointDerivatives",
                str(self.provides_adjoint_derivatives).lower(),
            )
        if (
            self.provides_per_element_dependencies is not None
            and self.provides_per_element_dependencies
        ):
            element.set(
                "providesPerElementDependencies",
                str(self.provides_per_element_dependencies).lower(),
            )
        if (
            self.can_handle_variable_communication_step_size is not None
            and self.can_handle_variable_communication_step_size
        ):
            element.set(
                "canHandleVariableCommunicationStepSize",
                str(self.can_handle_variable_communication_step_size).lower(),
            )
        if self.fixed_internal_step_size is not None:
            element.set("fixedInternalStepSize", str(self.fixed_internal_step_size))
        if (
            self.max_output_derivative_order is not None
            and self.max_output_derivative_order != 0
        ):
            element.set(
                "maxOutputDerivativeOrder", str(self.max_output_derivative_order)
            )
        if (
            self.recommended_intermediate_input_smoothness is not None
            and self.recommended_intermediate_input_smoothness != 0
        ):
            element.set(
                "recommendedIntermediateInputSmoothness",
                str(self.recommended_intermediate_input_smoothness),
            )
        if (
            self.provides_intermediate_update is not None
            and self.provides_intermediate_update
        ):
            element.set(
                "providesIntermediateUpdate",
                str(self.provides_intermediate_update).lower(),
            )
        if (
            self.might_return_early_from_do_step is not None
            and self.might_return_early_from_do_step
        ):
            element.set(
                "mightReturnEarlyFromDoStep",
                str(self.might_return_early_from_do_step).lower(),
            )
        if (
            self.can_return_early_after_intermediate_update is not None
            and self.can_return_early_after_intermediate_update
        ):
            element.set(
                "canReturnEarlyAfterIntermediateUpdate",
                str(self.can_return_early_after_intermediate_update).lower(),
            )
        if self.has_event_mode is not None and self.has_event_mode:
            element.set("hasEventMode", str(self.has_event_mode).lower())
        if (
            self.provides_evaluate_discrete_states is not None
            and self.provides_evaluate_discrete_states
        ):
            element.set(
                "providesEvaluateDiscreteStates",
                str(self.provides_evaluate_discrete_states).lower(),
            )
        return element


class ScheduledExecution(BaseModel):
    """Scheduled Execution interface definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    model_identifier: Annotated[str, Field(..., alias="modelIdentifier")]
    needs_execution_tool: Annotated[
        bool | None, Field(default=False, alias="needsExecutionTool")
    ] = False
    can_be_instantiated_only_once_per_process: Annotated[
        bool | None,
        Field(default=False, alias="canBeInstantiatedOnlyOncePerProcess"),
    ] = False
    can_get_and_set_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canGetAndSetFMUState")
    ] = False
    can_serialize_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canSerializeFMUState")
    ] = False
    provides_directional_derivatives: Annotated[
        bool | None, Field(default=False, alias="providesDirectionalDerivatives")
    ] = False
    provides_adjoint_derivatives: Annotated[
        bool | None, Field(default=False, alias="providesAdjointDerivatives")
    ] = False
    provides_per_element_dependencies: Annotated[
        bool | None, Field(default=False, alias="providesPerElementDependencies")
    ] = False

    def to_xml(self) -> Element:
        """Convert ScheduledExecution to XML Element"""
        element = Element("ScheduledExecution")
        element.set("modelIdentifier", self.model_identifier)
        if self.needs_execution_tool is not None and self.needs_execution_tool:
            element.set("needsExecutionTool", str(self.needs_execution_tool).lower())
        if (
            self.can_be_instantiated_only_once_per_process is not None
            and self.can_be_instantiated_only_once_per_process
        ):
            element.set(
                "canBeInstantiatedOnlyOncePerProcess",
                str(self.can_be_instantiated_only_once_per_process).lower(),
            )
        if (
            self.can_get_and_set_fmu_state is not None
            and self.can_get_and_set_fmu_state
        ):
            element.set(
                "canGetAndSetFMUState", str(self.can_get_and_set_fmu_state).lower()
            )
        if self.can_serialize_fmu_state is not None and self.can_serialize_fmu_state:
            element.set(
                "canSerializeFMUState", str(self.can_serialize_fmu_state).lower()
            )
        if (
            self.provides_directional_derivatives is not None
            and self.provides_directional_derivatives
        ):
            element.set(
                "providesDirectionalDerivatives",
                str(self.provides_directional_derivatives).lower(),
            )
        if (
            self.provides_adjoint_derivatives is not None
            and self.provides_adjoint_derivatives
        ):
            element.set(
                "providesAdjointDerivatives",
                str(self.provides_adjoint_derivatives).lower(),
            )
        if (
            self.provides_per_element_dependencies is not None
            and self.provides_per_element_dependencies
        ):
            element.set(
                "providesPerElementDependencies",
                str(self.provides_per_element_dependencies).lower(),
            )
        return element


class Category(BaseModel):
    """Log category definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[str, Field(..., alias="name")]
    description: Annotated[str | None, Field(default=None, alias="description")] = None

    def to_xml(self) -> Element:
        """Convert Category to XML Element"""
        element = Element("Category")
        element.set("name", self.name)
        if self.description is not None:
            element.set("description", self.description)
        return element


class LogCategories(BaseModel):
    """Log categories list"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    categories: Annotated[list[Category], Field(..., alias="Category")]

    def to_xml(self) -> Element:
        """Convert LogCategories to XML Element"""
        element = Element("LogCategories")
        if self.categories is not None:
            for category in self.categories:
                element.append(category.to_xml())
        return element


class DefaultExperiment(BaseModel):
    """Default experiment configuration"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start_time: Annotated[float | None, Field(default=None, alias="startTime")] = None
    stop_time: Annotated[float | None, Field(default=None, alias="stopTime")] = None
    tolerance: Annotated[float | None, Field(default=None, alias="tolerance")] = None
    step_size: Annotated[float | None, Field(default=None, alias="stepSize")] = None

    def to_xml(self) -> Element:
        """Convert DefaultExperiment to XML Element"""
        element = Element("DefaultExperiment")
        if self.start_time is not None:
            element.set("startTime", str(self.start_time))
        if self.stop_time is not None:
            element.set("stopTime", str(self.stop_time))
        if self.tolerance is not None:
            element.set("tolerance", str(self.tolerance))
        if self.step_size is not None:
            element.set("stepSize", str(self.step_size))
        return element


class Tool(BaseModel):
    """Tool-specific annotation"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[str, Field(..., alias="name")]
    # For simplicity, we'll use a generic dict for the content of the annotation
    # In a more complete implementation, this could be more structured
    content: Annotated[dict | None, Field(default=None)] = None

    def to_xml(self) -> Element:
        """Convert Tool to XML Element"""
        element = Element("Tool")
        element.set("name", self.name)
        # Note: content is not currently handled in XML conversion
        return element


class Annotation(BaseModel):
    """Vendor annotations"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    tools: Annotated[list[Tool] | None, Field(default=None, alias="Tool")] = None

    def to_xml(self) -> Element:
        """Convert Annotation to XML Element"""
        element = Element("Annotation")
        if self.tools is not None:
            for tool in self.tools:
                element.append(tool.to_xml())
        return element


class Dimension(BaseModel):
    """Array dimension definition for FMI 3.0 variables"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: Annotated[int | None, Field(default=None, alias="start")] = None
    value_reference: Annotated[
        int | None, Field(default=None, alias="valueReference")
    ] = None

    @model_validator(mode="after")
    def _validate_exclusive_start_or_value_reference(self) -> "Dimension":
        """Ensure exactly one of start or value_reference is provided."""
        if (self.start is None) == (self.value_reference is None):
            raise ValueError(
                "Dimension: exactly one of start or valueReference must be set"
            )
        return self

    def to_xml(self) -> Element:
        """Convert Dimension to XML Element"""
        element = Element("Dimension")
        if self.start is not None:
            element.set("start", str(self.start))
        if self.value_reference is not None:
            element.set("valueReference", str(self.value_reference))
        return element


class Unknown(BaseModel):
    """Unknown variable definition for model structure"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    value_reference: Annotated[int, Field(..., alias="valueReference")]
    dependencies: Annotated[
        list[int] | None, Field(default=None, alias="dependencies")
    ] = None
    dependencies_kind: Annotated[
        list[str] | None,
        Field(default=None, alias="dependenciesKind"),
    ] = None

    # Keep index as an alias for backward compatibility
    @property
    def index(self) -> int:
        return self.value_reference

    def to_xml(self) -> Element:
        """Convert Unknown to XML Element"""
        element = Element("Unknown")
        element.set("valueReference", str(self.value_reference))
        if self.dependencies is not None:
            element.set("dependencies", " ".join(map(str, self.dependencies)))
        if self.dependencies_kind is not None:
            element.set(
                "dependenciesKind",
                " ".join([str(kind) for kind in self.dependencies_kind]),
            )
        return element


class ModelStructure(BaseModel):
    """Model structure definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    outputs: Annotated[list[Unknown] | None, Field(default=None, alias="Output")] = None
    continuous_state_derivatives: Annotated[
        list[Unknown] | None, Field(default=None, alias="ContinuousStateDerivative")
    ] = None
    clocked_states: Annotated[
        list[Unknown] | None, Field(default=None, alias="ClockedState")
    ] = None
    initial_unknowns: Annotated[
        list[Unknown] | None, Field(default=None, alias="InitialUnknown")
    ] = None
    event_indicators: Annotated[
        list[Unknown] | None, Field(default=None, alias="EventIndicator")
    ] = None

    def to_xml(self) -> Element:
        """Convert ModelStructure to XML Element"""
        element = Element("ModelStructure")
        if self.outputs is not None:
            for output in self.outputs:
                output_elem = output.to_xml()
                output_elem.tag = "Output"
                element.append(output_elem)
        if self.continuous_state_derivatives is not None:
            for derivative in self.continuous_state_derivatives:
                derivative_elem = derivative.to_xml()
                derivative_elem.tag = "ContinuousStateDerivative"
                element.append(derivative_elem)
        if self.clocked_states is not None:
            for clocked_state in self.clocked_states:
                clocked_elem = clocked_state.to_xml()
                clocked_elem.tag = "ClockedState"
                element.append(clocked_elem)
        if self.initial_unknowns is not None:
            for initial_unknown in self.initial_unknowns:
                initial_elem = initial_unknown.to_xml()
                initial_elem.tag = "InitialUnknown"
                element.append(initial_elem)
        if self.event_indicators is not None:
            for event_indicator in self.event_indicators:
                event_elem = event_indicator.to_xml()
                event_elem.tag = "EventIndicator"
                element.append(event_elem)
        return element


# Variable definitions for FMI 3.0
class Float32Variable(BaseModel):
    """Float32 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    unit: Annotated[
        str | None,
        Field(default=None, alias="unit", description="Unit of the variable"),
    ] = None
    display_unit: Annotated[
        str | None,
        Field(
            default=None,
            alias="displayUnit",
            description='Default display unit, provided the conversion of values in "unit" to values in "displayUnit" is defined in UnitDefinitions / Unit / DisplayUnit.',
        ),
    ] = None
    relative_quantity: Annotated[
        bool | None,
        Field(
            default=False,
            alias="relativeQuantity",
            description="If relativeQuantity=true, offset for displayUnit must be ignored.",
        ),
    ] = False
    min_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    nominal: Annotated[
        float | None,
        Field(
            default=None,
            alias="nominal",
            description="Nominal value of the variable. nominal > 0.0 required",
        ),
    ] = None
    unbounded: Annotated[
        bool | None,
        Field(
            default=False,
            alias="unbounded",
            description="Set to true, e.g., for crank angle. If true and variable is a state, relative tolerance should be zero on this variable.",
        ),
    ] = False
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[float] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Float32Variable to XML Element"""
        element = Element("Float32")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.unit is not None:
            element.set("unit", self.unit)
        if self.display_unit is not None:
            element.set("displayUnit", self.display_unit)
        if self.relative_quantity is not None and self.relative_quantity:
            element.set("relativeQuantity", str(self.relative_quantity).lower())
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.nominal is not None:
            element.set("nominal", str(self.nominal))
        if self.unbounded is not None and self.unbounded:
            element.set("unbounded", str(self.unbounded).lower())
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class Float64Variable(BaseModel):
    """Float64 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    unit: Annotated[
        str | None,
        Field(default=None, alias="unit", description="Unit of the variable"),
    ] = None
    display_unit: Annotated[
        str | None,
        Field(
            default=None,
            alias="displayUnit",
            description='Default display unit, provided the conversion of values in "unit" to values in "displayUnit" is defined in UnitDefinitions / Unit / DisplayUnit.',
        ),
    ] = None
    relative_quantity: Annotated[
        bool | None,
        Field(
            default=False,
            alias="relativeQuantity",
            description="If relativeQuantity=true, offset for displayUnit must be ignored.",
        ),
    ] = False
    min_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    nominal: Annotated[
        float | None,
        Field(
            default=None,
            alias="nominal",
            description="Nominal value of the variable. nominal > 0.0 required",
        ),
    ] = None
    unbounded: Annotated[
        bool | None,
        Field(
            default=False,
            alias="unbounded",
            description="Set to true, e.g., for crank angle. If true and variable is a state, relative tolerance should be zero on this variable.",
        ),
    ] = False
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[float] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Float64Variable to XML Element"""
        element = Element("Float64")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.unit is not None:
            element.set("unit", self.unit)
        if self.display_unit is not None:
            element.set("displayUnit", self.display_unit)
        if self.relative_quantity is not None and self.relative_quantity:
            element.set("relativeQuantity", str(self.relative_quantity).lower())
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.nominal is not None:
            element.set("nominal", str(self.nominal))
        if self.unbounded is not None and self.unbounded:
            element.set("unbounded", str(self.unbounded).lower())
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class Int8Variable(BaseModel):
    """Int8 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Int8Variable to XML Element"""
        element = Element("Int8")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class Int16Variable(BaseModel):
    """Int16 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Int16Variable to XML Element"""
        element = Element("Int16")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class Int32Variable(BaseModel):
    """Int32 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Int32Variable to XML Element"""
        element = Element("Int32")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class Int64Variable(BaseModel):
    """Int64 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert Int64Variable to XML Element"""
        element = Element("Int64")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class UInt8Variable(BaseModel):
    """UInt8 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert UInt8Variable to XML Element"""
        element = Element("UInt8")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class UInt16Variable(BaseModel):
    """UInt16 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert UInt16Variable to XML Element"""
        element = Element("UInt16")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class UInt32Variable(BaseModel):
    """UInt32 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert UInt32Variable to XML Element"""
        element = Element("UInt32")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class UInt64Variable(BaseModel):
    """UInt64 variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert UInt64Variable to XML Element"""
        element = Element("UInt64")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class BooleanVariable(BaseModel):
    """Boolean variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[bool] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert BooleanVariable to XML Element"""
        element = Element("Boolean")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join([str(s).lower() for s in self.start]))
        return element


class StringVariable(BaseModel):
    """String variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[str] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert StringVariable to XML Element"""
        element = Element("String")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(self.start))
        return element


class BinaryVariable(BaseModel):
    """Binary variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    mime_type: Annotated[
        str | None,
        Field(
            default="application/octet-stream",
            alias="mimeType",
            description="MIME type for binary data",
        ),
    ] = "application/octet-stream"
    max_size: Annotated[
        int | None,
        Field(
            default=None,
            alias="maxSize",
            description="Maximum size of binary data",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[bytes] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert BinaryVariable to XML Element"""
        element = Element("Binary")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.mime_type is not None and self.mime_type != "application/octet-stream":
            element.set("mimeType", self.mime_type)
        if self.max_size is not None:
            element.set("maxSize", str(self.max_size))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join([s.hex() for s in self.start]))
        return element


class EnumerationVariable(BaseModel):
    """Enumeration variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    variability: Annotated[
        VariabilityEnum | None,
        Field(
            default=VariabilityEnum.continuous,
            alias="variability",
            description="constant: value never changes; fixed: value fixed after initialization; tunable: value constant between external events; discrete: value constant between internal events; continuous: no restriction on value changes",
        ),
    ] = VariabilityEnum.continuous
    initial: Annotated[
        InitialEnum | None,
        Field(
            default=None,
            alias="initial",
            description="exact: initialized with start value; approx: iteration variable that starts with start value; calculated: calculated from other variables. If not provided, initial is deduced from causality and variability (details see specification)",
        ),
    ] = None
    can_handle_multiple_set_per_time_instant: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canHandleMultipleSetPerTimeInstant",
            description='Only for variables with variability = "continuous": If present with value = false, then only one fmi3SetXXX call is allowed at one super dense time instant.',
        ),
    ] = None
    declared_type: Annotated[
        str,
        Field(
            ...,
            alias="declaredType",
            description="Name of type defined with TypeDefinitions / Type",
        ),
    ]
    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    dimensions: Annotated[
        list[Dimension] | None,
        Field(
            default=None,
            alias="Dimension",
            description="Array dimensions of the variable",
        ),
    ] = None
    start: Annotated[
        list[int] | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert EnumerationVariable to XML Element"""
        element = Element("Enumeration")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        element.set("declaredType", self.declared_type)
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.variability is not None:
            element.set("variability", self.variability.value)
        if self.initial is not None:
            element.set("initial", self.initial.value)
        if self.can_handle_multiple_set_per_time_instant is not None:
            element.set(
                "canHandleMultipleSetPerTimeInstant",
                str(self.can_handle_multiple_set_per_time_instant).lower(),
            )
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.dimensions is not None:
            for dim in self.dimensions:
                element.append(dim.to_xml())
        if self.start is not None:
            element.set("start", " ".join(map(str, self.start)))
        return element


class ClockVariable(BaseModel):
    """Clock variable definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Identifier of variable, e.g. "a.b.mod[3,4].\'#123\'.c". "name" must be unique with respect to all other elements of the ModelVariables list',
        ),
    ]
    value_reference: Annotated[
        int,
        Field(
            ...,
            alias="valueReference",
            description="Identifier for variable value in FMI3 function calls (not necessarily unique with respect to all variables)",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the variable"
        ),
    ] = None
    causality: Annotated[
        CausalityEnum | None,
        Field(
            default=CausalityEnum.local,
            alias="causality",
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time); structuralParameter: structural parameter",
        ),
    ] = CausalityEnum.local
    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / Type providing defaults.",
        ),
    ] = None
    can_be_deactivated: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canBeDeactivated",
            description="Whether the clock can be deactivated",
        ),
    ] = None
    priority: Annotated[
        int | None,
        Field(
            default=None,
            alias="priority",
            description="Priority of the clock",
        ),
    ] = None
    synchronised: Annotated[
        bool | None,
        Field(
            default=None,
            alias="synchronised",
            description="Whether the clock is synchronized",
        ),
    ] = None
    support_frequency_resolution: Annotated[
        float | None,
        Field(
            default=None,
            alias="supportFrequencyResolution",
            description="Support frequency resolution",
        ),
    ] = None
    support_tentative_steps: Annotated[
        bool | None,
        Field(
            default=None,
            alias="supportTentativeSteps",
            description="Whether tentative steps are supported",
        ),
    ] = None
    resolution: Annotated[
        int | None,
        Field(
            default=None,
            alias="resolution",
            description="Resolution of the clock",
        ),
    ] = None
    interval_counter: Annotated[
        int | None,
        Field(
            default=None,
            alias="intervalCounter",
            description="Interval counter",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert ClockVariable to XML Element"""
        element = Element("Clock")
        element.set("name", self.name)
        element.set("valueReference", str(self.value_reference))
        if self.description is not None:
            element.set("description", self.description)
        if self.causality is not None:
            element.set("causality", self.causality.value)
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.can_be_deactivated is not None:
            element.set("canBeDeactivated", str(self.can_be_deactivated).lower())
        if self.priority is not None:
            element.set("priority", str(self.priority))
        if self.synchronised is not None:
            element.set("synchronised", str(self.synchronised).lower())
        if self.support_frequency_resolution is not None:
            element.set(
                "supportFrequencyResolution", str(self.support_frequency_resolution)
            )
        if self.support_tentative_steps is not None:
            element.set(
                "supportTentativeSteps", str(self.support_tentative_steps).lower()
            )
        if self.resolution is not None:
            element.set("resolution", str(self.resolution))
        if self.interval_counter is not None:
            element.set("intervalCounter", str(self.interval_counter))
        return element


class Variable(BaseModel):
    """Union of all variable types for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    # Variable type (one of these should be present)
    float32: Annotated[
        Optional[Float32Variable],
        Field(default=None, alias="Float32", description="Float32 variable definition"),
    ] = None
    float64: Annotated[
        Optional[Float64Variable],
        Field(default=None, alias="Float64", description="Float64 variable definition"),
    ] = None
    int8: Annotated[
        Optional[Int8Variable],
        Field(default=None, alias="Int8", description="Int8 variable definition"),
    ] = None
    int16: Annotated[
        Optional[Int16Variable],
        Field(default=None, alias="Int16", description="Int16 variable definition"),
    ] = None
    int32: Annotated[
        Optional[Int32Variable],
        Field(default=None, alias="Int32", description="Int32 variable definition"),
    ] = None
    int64: Annotated[
        Optional[Int64Variable],
        Field(default=None, alias="Int64", description="Int64 variable definition"),
    ] = None
    uint8: Annotated[
        Optional[UInt8Variable],
        Field(default=None, alias="UInt8", description="UInt8 variable definition"),
    ] = None
    uint16: Annotated[
        Optional[UInt16Variable],
        Field(default=None, alias="UInt16", description="UInt16 variable definition"),
    ] = None
    uint32: Annotated[
        Optional[UInt32Variable],
        Field(default=None, alias="UInt32", description="UInt32 variable definition"),
    ] = None
    uint64: Annotated[
        Optional[UInt64Variable],
        Field(default=None, alias="UInt64", description="UInt64 variable definition"),
    ] = None
    boolean: Annotated[
        Optional[BooleanVariable],
        Field(default=None, alias="Boolean", description="Boolean variable definition"),
    ] = None
    string: Annotated[
        Optional[StringVariable],
        Field(default=None, alias="String", description="String variable definition"),
    ] = None
    binary: Annotated[
        Optional[BinaryVariable],
        Field(default=None, alias="Binary", description="Binary variable definition"),
    ] = None
    enumeration: Annotated[
        Optional[EnumerationVariable],
        Field(
            default=None,
            alias="Enumeration",
            description="Enumeration variable definition",
        ),
    ] = None
    clock: Annotated[
        Optional[ClockVariable],
        Field(default=None, alias="Clock", description="Clock variable definition"),
    ] = None

    def get_variable_type(self) -> str | None:
        """Get the type of variable based on which field is set"""
        if self.float32 is not None:
            return "Float32"
        elif self.float64 is not None:
            return "Float64"
        elif self.int8 is not None:
            return "Int8"
        elif self.int16 is not None:
            return "Int16"
        elif self.int32 is not None:
            return "Int32"
        elif self.int64 is not None:
            return "Int64"
        elif self.uint8 is not None:
            return "UInt8"
        elif self.uint16 is not None:
            return "UInt16"
        elif self.uint32 is not None:
            return "UInt32"
        elif self.uint64 is not None:
            return "UInt64"
        elif self.boolean is not None:
            return "Boolean"
        elif self.string is not None:
            return "String"
        elif self.binary is not None:
            return "Binary"
        elif self.enumeration is not None:
            return "Enumeration"
        elif self.clock is not None:
            return "Clock"
        return None

    def _concrete(
        self,
    ) -> (
        Float32Variable
        | Float64Variable
        | Int8Variable
        | Int16Variable
        | Int32Variable
        | Int64Variable
        | UInt8Variable
        | UInt16Variable
        | UInt32Variable
        | UInt64Variable
        | BooleanVariable
        | StringVariable
        | BinaryVariable
        | EnumerationVariable
        | ClockVariable
    ):
        """Return the underlying concrete variable instance or raise if unset."""
        for attr in (
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "boolean",
            "string",
            "binary",
            "enumeration",
            "clock",
        ):
            value = getattr(self, attr)
            if value is not None:
                return value
        raise ValueError("No variable type is set")

    @property
    def concrete(
        self,
    ) -> (
        Float32Variable
        | Float64Variable
        | Int8Variable
        | Int16Variable
        | Int32Variable
        | Int64Variable
        | UInt8Variable
        | UInt16Variable
        | UInt32Variable
        | UInt64Variable
        | BooleanVariable
        | StringVariable
        | BinaryVariable
        | EnumerationVariable
        | ClockVariable
    ):
        """Direct access to the concrete variable instance."""
        return self._concrete()

    @property
    def name(self) -> str:
        return self.concrete.name

    @property
    def value_reference(self) -> int:
        return self.concrete.value_reference

    @property
    def causality(self) -> CausalityEnum | None:
        return getattr(self.concrete, "causality", None)

    @property
    def variability(self) -> VariabilityEnum | None:
        return getattr(self.concrete, "variability", None)

    @property
    def initial(self) -> InitialEnum | None:
        return getattr(self.concrete, "initial", None)

    def to_xml(self) -> Element:
        """Convert Variable to XML Element"""
        # Add the appropriate variable type element
        if self.float32 is not None:
            return self.float32.to_xml()
        elif self.float64 is not None:
            return self.float64.to_xml()
        elif self.int8 is not None:
            return self.int8.to_xml()
        elif self.int16 is not None:
            return self.int16.to_xml()
        elif self.int32 is not None:
            return self.int32.to_xml()
        elif self.int64 is not None:
            return self.int64.to_xml()
        elif self.uint8 is not None:
            return self.uint8.to_xml()
        elif self.uint16 is not None:
            return self.uint16.to_xml()
        elif self.uint32 is not None:
            return self.uint32.to_xml()
        elif self.uint64 is not None:
            return self.uint64.to_xml()
        elif self.boolean is not None:
            return self.boolean.to_xml()
        elif self.string is not None:
            return self.string.to_xml()
        elif self.binary is not None:
            return self.binary.to_xml()
        elif self.enumeration is not None:
            return self.enumeration.to_xml()
        elif self.clock is not None:
            return self.clock.to_xml()
        else:
            raise ValueError("No variable type is set")


class UnitDefinitions(BaseModel):
    """Unit definitions list for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    units: Annotated[
        list[Unit],
        Field(
            ...,
            alias="Unit",
            description="Unit definition (with respect to SI base units) and default display units",
        ),
    ]

    def to_xml(self) -> Element:
        """Convert UnitDefinitions to XML Element"""
        element = Element("UnitDefinitions")
        if self.units is not None:
            for unit in self.units:
                element.append(unit.to_xml())
        return element


class TypeDefinitions(BaseModel):
    """Type definitions list for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    types: Annotated[list[Type], Field(..., alias="Type")]

    def to_xml(self) -> Element:
        """Convert TypeDefinitions to XML Element"""
        element = Element("TypeDefinitions")
        if self.types is not None:
            for type_def in self.types:
                element.append(type_def.to_xml())
        return element


class FmiModelDescription(BaseModel):
    """Main FMI 3.0 model description"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    fmi_version: Annotated[
        str,
        Field(
            ...,
            alias="fmiVersion",
            description="Version of FMI (for FMI 3.x.x)",
        ),
    ]
    model_name: Annotated[
        str,
        Field(
            ...,
            alias="modelName",
            description='Class name of FMU, e.g. "A.B.C" (several FMU instances are possible)',
        ),
    ]
    instantiation_token: Annotated[
        str,
        Field(
            ...,
            alias="instantiationToken",
            description="Fingerprint of xml-file content to verify that xml-file and C-functions are compatible to each other",
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None, alias="description", description="Description of the model"
        ),
    ] = None
    author: Annotated[
        str | None,
        Field(default=None, alias="author", description="Author of the model"),
    ] = None
    version: Annotated[
        str | None,
        Field(
            default=None, alias="version", description='Version of FMU, e.g., "1.4.1"'
        ),
    ] = None
    copyright: Annotated[
        str | None,
        Field(
            default=None,
            alias="copyright",
            description='Information on intellectual property copyright for this FMU, such as "© MyCompany 2011"',
        ),
    ] = None
    license: Annotated[
        str | None,
        Field(
            default=None,
            alias="license",
            description='Information on intellectual property licensing for this FMU, such as "BSD license", "Proprietary", or "Public Domain"',
        ),
    ] = None
    generation_tool: Annotated[
        str | None,
        Field(
            default=None,
            alias="generationTool",
            description="Tool that generated the FMU",
        ),
    ] = None
    generation_date_and_time: Annotated[
        str | None,
        Field(
            default=None,
            alias="generationDateAndTime",
            description="Date and time when the FMU was generated",
        ),
    ] = None
    variable_naming_convention: Annotated[
        VariableNamingConventionEnum | None,
        Field(
            default=VariableNamingConventionEnum.flat,
            alias="variableNamingConvention",
            description="Naming convention for variables: flat or structured",
        ),
    ] = VariableNamingConventionEnum.flat

    # Optional components
    model_exchange: Annotated[
        ModelExchange | None,
        Field(
            default=None,
            alias="ModelExchange",
            description="Model Exchange interface definition",
        ),
    ] = None
    co_simulation: Annotated[
        CoSimulation | None,
        Field(
            default=None,
            alias="CoSimulation",
            description="Co-Simulation interface definition",
        ),
    ] = None
    scheduled_execution: Annotated[
        ScheduledExecution | None,
        Field(
            default=None,
            alias="ScheduledExecution",
            description="Scheduled Execution interface definition",
        ),
    ] = None
    unit_definitions: Annotated[
        UnitDefinitions | None,
        Field(
            default=None,
            alias="UnitDefinitions",
            description="Unit definitions for the model",
        ),
    ] = None
    type_definitions: Annotated[
        TypeDefinitions | None,
        Field(
            default=None,
            alias="TypeDefinitions",
            description="Type definitions for the model",
        ),
    ] = None
    log_categories: Annotated[
        LogCategories | None,
        Field(
            default=None,
            alias="LogCategories",
            description="Log categories available in FMU",
        ),
    ] = None
    default_experiment: Annotated[
        DefaultExperiment | None,
        Field(
            default=None,
            alias="DefaultExperiment",
            description="Default experiment configuration",
        ),
    ] = None
    vendor_annotations: Annotated[
        Annotation | None,
        Field(
            default=None,
            alias="VendorAnnotations",
            description="Tool specific data (ignored by other tools)",
        ),
    ] = None
    model_variables: Annotated[
        list[Variable],
        Field(
            ...,
            alias="ModelVariables",
            description="Ordered list of all variables (first definition has index = 1)",
        ),
    ]
    model_structure: Annotated[
        ModelStructure | None,
        Field(
            default=None,
            alias="ModelStructure",
            description="Ordered lists of outputs, exposed state derivatives, and the initial unknowns. Optionally, the functional dependency of these variables can be defined.",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert FmiModelDescription to XML Element"""
        element = Element("fmiModelDescription")
        element.set("fmiVersion", self.fmi_version)
        element.set("modelName", self.model_name)
        element.set("instantiationToken", self.instantiation_token)
        if self.description is not None:
            element.set("description", self.description)
        if self.author is not None:
            element.set("author", self.author)
        if self.version is not None:
            element.set("version", self.version)
        if self.copyright is not None:
            element.set("copyright", self.copyright)
        if self.license is not None:
            element.set("license", self.license)
        if self.generation_tool is not None:
            element.set("generationTool", self.generation_tool)
        if self.generation_date_and_time is not None:
            element.set("generationDateAndTime", self.generation_date_and_time)
        if (
            self.variable_naming_convention is not None
            and self.variable_naming_convention != VariableNamingConventionEnum.flat
        ):
            element.set(
                "variableNamingConvention", self.variable_naming_convention.value
            )

        # Add optional components
        if self.model_exchange is not None:
            element.append(self.model_exchange.to_xml())
        if self.co_simulation is not None:
            element.append(self.co_simulation.to_xml())
        if self.scheduled_execution is not None:
            element.append(self.scheduled_execution.to_xml())
        if self.unit_definitions is not None:
            element.append(self.unit_definitions.to_xml())
        if self.type_definitions is not None:
            element.append(self.type_definitions.to_xml())
        if self.log_categories is not None:
            element.append(self.log_categories.to_xml())
        if self.default_experiment is not None:
            element.append(self.default_experiment.to_xml())
        if self.vendor_annotations is not None:
            element.append(self.vendor_annotations.to_xml())
        if self.model_variables is not None:
            model_vars_elem = Element("ModelVariables")
            for variable in self.model_variables:
                model_vars_elem.append(variable.to_xml())
            element.append(model_vars_elem)
        if self.model_structure is not None:
            element.append(self.model_structure.to_xml())

        return element


class Float32Type(BaseModel):
    """Float32 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    unit: Annotated[
        str | None,
        Field(default=None, alias="unit", description="Unit of the variable"),
    ] = None
    display_unit: Annotated[
        str | None,
        Field(
            default=None,
            alias="displayUnit",
            description='Default display unit, provided the conversion of values in "unit" to values in "displayUnit" is defined in UnitDefinitions / Unit / DisplayUnit.',
        ),
    ] = None
    relative_quantity: Annotated[
        bool | None,
        Field(
            default=False,
            alias="relativeQuantity",
            description="If relativeQuantity=true, offset for displayUnit must be ignored.",
        ),
    ] = False
    min_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    nominal: Annotated[
        float | None,
        Field(
            default=None,
            alias="nominal",
            description="Nominal value of the variable. nominal > 0.0 required",
        ),
    ] = None
    unbounded: Annotated[
        bool | None,
        Field(
            default=False,
            alias="unbounded",
            description="Set to true, e.g., for crank angle. If true and variable is a state, relative tolerance should be zero on this variable.",
        ),
    ] = False

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"Float32Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert Float32Type to XML Element"""
        element = Element("Float32")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.unit is not None:
            element.set("unit", self.unit)
        if self.display_unit is not None:
            element.set("displayUnit", self.display_unit)
        if self.relative_quantity is not None and self.relative_quantity:
            element.set("relativeQuantity", str(self.relative_quantity).lower())
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.nominal is not None:
            element.set("nominal", str(self.nominal))
        if self.unbounded is not None and self.unbounded:
            element.set("unbounded", str(self.unbounded).lower())
        return element


class Float64Type(BaseModel):
    """Float64 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    unit: Annotated[
        str | None,
        Field(default=None, alias="unit", description="Unit of the variable"),
    ] = None
    display_unit: Annotated[
        str | None,
        Field(
            default=None,
            alias="displayUnit",
            description='Default display unit, provided the conversion of values in "unit" to values in "displayUnit" is defined in UnitDefinitions / Unit / DisplayUnit.',
        ),
    ] = None
    relative_quantity: Annotated[
        bool | None,
        Field(
            default=False,
            alias="relativeQuantity",
            description="If relativeQuantity=true, offset for displayUnit must be ignored.",
        ),
    ] = False
    min_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        float | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None
    nominal: Annotated[
        float | None,
        Field(
            default=None,
            alias="nominal",
            description="Nominal value of the variable. nominal > 0.0 required",
        ),
    ] = None
    unbounded: Annotated[
        bool | None,
        Field(
            default=False,
            alias="unbounded",
            description="Set to true, e.g., for crank angle. If true and variable is a state, relative tolerance should be zero on this variable.",
        ),
    ] = False

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"Float64Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert Float64Type to XML Element"""
        element = Element("Float64")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.unit is not None:
            element.set("unit", self.unit)
        if self.display_unit is not None:
            element.set("displayUnit", self.display_unit)
        if self.relative_quantity is not None and self.relative_quantity:
            element.set("relativeQuantity", str(self.relative_quantity).lower())
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.nominal is not None:
            element.set("nominal", str(self.nominal))
        if self.unbounded is not None and self.unbounded:
            element.set("unbounded", str(self.unbounded).lower())
        return element


class Int8Type(BaseModel):
    """Int8 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"Int8Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert Int8Type to XML Element"""
        element = Element("Int8")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class Int16Type(BaseModel):
    """Int16 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"Int16Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert Int16Type to XML Element"""
        element = Element("Int16")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class Int32Type(BaseModel):
    """Int32 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"Int32Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert Int32Type to XML Element"""
        element = Element("Int32")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class Int64Type(BaseModel):
    """Int64 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"Int64Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert Int64Type to XML Element"""
        element = Element("Int64")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class UInt8Type(BaseModel):
    """UInt8 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"UInt8Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert UInt8Type to XML Element"""
        element = Element("UInt8")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class UInt16Type(BaseModel):
    """UInt16 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"UInt16Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert UInt16Type to XML Element"""
        element = Element("UInt16")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class UInt32Type(BaseModel):
    """UInt32 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"UInt32Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert UInt32Type to XML Element"""
        element = Element("UInt32")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class UInt64Type(BaseModel):
    """UInt64 type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    min_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="min",
            description="Minimum value of the variable. max >= min required",
        ),
    ] = None
    max_value: Annotated[
        int | None,
        Field(
            default=None,
            alias="max",
            description="Maximum value of the variable. max >= min required",
        ),
    ] = None

    @model_validator(mode="after")
    def check_min_max(self):
        """Validator to check that max >= min if both are set"""
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError(
                    f"UInt64Type: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert UInt64Type to XML Element"""
        element = Element("UInt64")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class BooleanType(BaseModel):
    """Boolean type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    pass  # Boolean types have no specific attributes

    def to_xml(self) -> Element:
        """Convert BooleanType to XML Element"""
        element = Element("Boolean")
        return element


class StringType(BaseModel):
    """String type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    pass  # String types have no specific attributes

    def to_xml(self) -> Element:
        """Convert StringType to XML Element"""
        element = Element("String")
        return element


class BinaryType(BaseModel):
    """Binary type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    mime_type: Annotated[
        str | None,
        Field(
            default="application/octet-stream",
            alias="mimeType",
            description="MIME type for binary data",
        ),
    ] = "application/octet-stream"
    max_size: Annotated[
        int | None,
        Field(
            default=None,
            alias="maxSize",
            description="Maximum size of binary data",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert BinaryType to XML Element"""
        element = Element("Binary")
        if self.mime_type is not None and self.mime_type != "application/octet-stream":
            element.set("mimeType", self.mime_type)
        if self.max_size is not None:
            element.set("maxSize", str(self.max_size))
        return element


class EnumerationType(BaseModel):
    """Enumeration type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    quantity: Annotated[
        str | None,
        Field(
            default=None,
            alias="quantity",
            description="Physical quantity of the variable",
        ),
    ] = None
    items: Annotated[
        list[Item], Field(..., alias="Item", description="List of enumeration items")
    ]

    def to_xml(self) -> Element:
        """Convert EnumerationType to XML Element"""
        element = Element("Enumeration")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.items is not None:
            for item in self.items:
                element.append(item.to_xml())
        return element


class ClockType(BaseModel):
    """Clock type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    can_be_deactivated: Annotated[
        bool | None,
        Field(
            default=None,
            alias="canBeDeactivated",
            description="Whether the clock can be deactivated",
        ),
    ] = None
    priority: Annotated[
        int | None,
        Field(
            default=None,
            alias="priority",
            description="Priority of the clock",
        ),
    ] = None
    synchronised: Annotated[
        bool | None,
        Field(
            default=None,
            alias="synchronised",
            description="Whether the clock is synchronized",
        ),
    ] = None
    support_frequency_resolution: Annotated[
        float | None,
        Field(
            default=None,
            alias="supportFrequencyResolution",
            description="Support frequency resolution",
        ),
    ] = None
    support_tentative_steps: Annotated[
        bool | None,
        Field(
            default=None,
            alias="supportTentativeSteps",
            description="Whether tentative steps are supported",
        ),
    ] = None
    resolution: Annotated[
        int | None,
        Field(
            default=None,
            alias="resolution",
            description="Resolution of the clock",
        ),
    ] = None
    interval_counter: Annotated[
        int | None,
        Field(
            default=None,
            alias="intervalCounter",
            description="Interval counter",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert ClockType to XML Element"""
        element = Element("Clock")
        if self.can_be_deactivated is not None:
            element.set("canBeDeactivated", str(self.can_be_deactivated).lower())
        if self.priority is not None:
            element.set("priority", str(self.priority))
        if self.synchronised is not None:
            element.set("synchronised", str(self.synchronised).lower())
        if self.support_frequency_resolution is not None:
            element.set(
                "supportFrequencyResolution", str(self.support_frequency_resolution)
            )
        if self.support_tentative_steps is not None:
            element.set(
                "supportTentativeSteps", str(self.support_tentative_steps).lower()
            )
        if self.resolution is not None:
            element.set("resolution", str(self.resolution))
        if self.interval_counter is not None:
            element.set("intervalCounter", str(self.interval_counter))
        return element


class Type(BaseModel):
    """Type definition for FMI 3.0"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Name of Type element. "name" must be unique with respect to all other elements of the TypeDefinitions list. Furthermore, "name" of a Type must be different to all "name"s of Variable.',
        ),
    ]
    description: Annotated[
        str | None,
        Field(alias="description", description="Description of the Type"),
    ] = None

    # Type-specific definitions (only one should be present)
    float32: Annotated[
        Optional[Float32Type],
        Field(alias="Float32", description="Float32 type definition"),
    ] = None
    float64: Annotated[
        Optional[Float64Type],
        Field(alias="Float64", description="Float64 type definition"),
    ] = None
    int8: Annotated[
        Optional[Int8Type],
        Field(alias="Int8", description="Int8 type definition"),
    ] = None
    int16: Annotated[
        Optional[Int16Type],
        Field(alias="Int16", description="Int16 type definition"),
    ] = None
    int32: Annotated[
        Optional[Int32Type],
        Field(alias="Int32", description="Int32 type definition"),
    ] = None
    int64: Annotated[
        Optional[Int64Type],
        Field(alias="Int64", description="Int64 type definition"),
    ] = None
    uint8: Annotated[
        Optional[UInt8Type],
        Field(alias="UInt8", description="UInt8 type definition"),
    ] = None
    uint16: Annotated[
        Optional[UInt16Type],
        Field(alias="UInt16", description="UInt16 type definition"),
    ] = None
    uint32: Annotated[
        Optional[UInt32Type],
        Field(alias="UInt32", description="UInt32 type definition"),
    ] = None
    uint64: Annotated[
        Optional[UInt64Type],
        Field(alias="UInt64", description="UInt64 type definition"),
    ] = None
    boolean: Annotated[
        Optional[BooleanType],
        Field(alias="Boolean", description="Boolean type definition"),
    ] = None
    string: Annotated[
        Optional[StringType],
        Field(alias="String", description="String type definition"),
    ] = None
    binary: Annotated[
        Optional[BinaryType],
        Field(alias="Binary", description="Binary type definition"),
    ] = None
    enumeration: Annotated[
        Optional[EnumerationType],
        Field(alias="Enumeration", description="Enumeration type definition"),
    ] = None
    clock: Annotated[
        Optional[ClockType],
        Field(alias="Clock", description="Clock type definition"),
    ] = None

    def get_type_category(self):
        """Get the type category based on which field is set"""
        if self.float32 is not None:
            return "Float32"
        elif self.float64 is not None:
            return "Float64"
        elif self.int8 is not None:
            return "Int8"
        elif self.int16 is not None:
            return "Int16"
        elif self.int32 is not None:
            return "Int32"
        elif self.int64 is not None:
            return "Int64"
        elif self.uint8 is not None:
            return "UInt8"
        elif self.uint16 is not None:
            return "UInt16"
        elif self.uint32 is not None:
            return "UInt32"
        elif self.uint64 is not None:
            return "UInt64"
        elif self.boolean is not None:
            return "Boolean"
        elif self.string is not None:
            return "String"
        elif self.binary is not None:
            return "Binary"
        elif self.enumeration is not None:
            return "Enumeration"
        elif self.clock is not None:
            return "Clock"
        return None

    def to_xml(self) -> Element:
        """Convert Type to XML Element"""
        element = Element("Type")
        element.set("name", self.name)
        if self.description is not None:
            element.set("description", self.description)

        # Add the appropriate type element
        if self.float32 is not None:
            element.append(self.float32.to_xml())
        elif self.float64 is not None:
            element.append(self.float64.to_xml())
        elif self.int8 is not None:
            element.append(self.int8.to_xml())
        elif self.int16 is not None:
            element.append(self.int16.to_xml())
        elif self.int32 is not None:
            element.append(self.int32.to_xml())
        elif self.int64 is not None:
            element.append(self.int64.to_xml())
        elif self.uint8 is not None:
            element.append(self.uint8.to_xml())
        elif self.uint16 is not None:
            element.append(self.uint16.to_xml())
        elif self.uint32 is not None:
            element.append(self.uint32.to_xml())
        elif self.uint64 is not None:
            element.append(self.uint64.to_xml())
        elif self.boolean is not None:
            element.append(self.boolean.to_xml())
        elif self.string is not None:
            element.append(self.string.to_xml())
        elif self.binary is not None:
            element.append(self.binary.to_xml())
        elif self.enumeration is not None:
            element.append(self.enumeration.to_xml())
        elif self.clock is not None:
            element.append(self.clock.to_xml())

        return element


def _parse_xml_to_model(xml_content: str | Element) -> FmiModelDescription:
    """
    Parse XML content and convert it to FmiModelDescription Pydantic model.

    Args:
        xml_content: XML string or ElementTree Element to parse

    Returns:
        FmiModelDescription: Parsed model instance
    """
    if isinstance(xml_content, str):
        root = ET.fromstring(xml_content)
    else:
        root = xml_content

    # Extract root attributes
    fmi_version = root.get("fmiVersion")
    if fmi_version is None:
        raise ValueError("fmiVersion attribute is required")
    model_name = root.get("modelName")
    if model_name is None:
        raise ValueError("modelName attribute is required")
    instantiation_token = root.get("instantiationToken")
    if instantiation_token is None:
        raise ValueError("instantiationToken attribute is required")
    description = root.get("description")
    author = root.get("author")
    version = root.get("version")
    copyright = root.get("copyright")
    license = root.get("license")
    generation_tool = root.get("generationTool")
    generation_date_and_time = root.get("generationDateAndTime")
    variable_naming_convention_str = root.get("variableNamingConvention")
    variable_naming_convention = (
        VariableNamingConventionEnum(variable_naming_convention_str)
        if variable_naming_convention_str
        else VariableNamingConventionEnum.flat
    )

    # Parse ModelExchange if present
    model_exchange_elem = root.find("ModelExchange")
    model_exchange = None
    if model_exchange_elem is not None:
        model_exchange = _parse_model_exchange(model_exchange_elem)

    # Parse CoSimulation if present
    co_simulation_elem = root.find("CoSimulation")
    co_simulation = None
    if co_simulation_elem is not None:
        co_simulation = _parse_co_simulation(co_simulation_elem)

    # Parse ScheduledExecution if present
    scheduled_execution_elem = root.find("ScheduledExecution")
    scheduled_execution = None
    if scheduled_execution_elem is not None:
        scheduled_execution = _parse_scheduled_execution(scheduled_execution_elem)

    # Parse UnitDefinitions if present
    unit_definitions_elem = root.find("UnitDefinitions")
    unit_definitions = None
    if unit_definitions_elem is not None:
        unit_definitions = _parse_unit_definitions(unit_definitions_elem)

    # Parse TypeDefinitions if present
    type_definitions_elem = root.find("TypeDefinitions")
    type_definitions = None
    if type_definitions_elem is not None:
        type_definitions = _parse_type_definitions(type_definitions_elem)

    # Parse LogCategories if present
    log_categories_elem = root.find("LogCategories")
    log_categories = None
    if log_categories_elem is not None:
        log_categories = _parse_log_categories(log_categories_elem)

    # Parse DefaultExperiment if present
    default_experiment_elem = root.find("DefaultExperiment")
    default_experiment = None
    if default_experiment_elem is not None:
        default_experiment = _parse_default_experiment(default_experiment_elem)

    # Parse VendorAnnotations if present
    vendor_annotations_elem = root.find("VendorAnnotations")
    vendor_annotations = None
    if vendor_annotations_elem is not None:
        vendor_annotations = _parse_vendor_annotations(vendor_annotations_elem)

    # Parse ModelVariables (required)
    model_variables_elem = root.find("ModelVariables")
    if model_variables_elem is None:
        raise ValueError("ModelVariables element is required")
    model_variables = _parse_model_variables(model_variables_elem)

    # Parse ModelStructure if present
    model_structure_elem = root.find("ModelStructure")
    model_structure = None
    if model_structure_elem is not None:
        model_structure = _parse_model_structure(model_structure_elem)

    return FmiModelDescription(
        fmi_version=fmi_version,
        model_name=model_name,
        instantiation_token=instantiation_token,
        description=description,
        author=author,
        version=version,
        copyright=copyright,
        license=license,
        generation_tool=generation_tool,
        generation_date_and_time=generation_date_and_time,
        variable_naming_convention=variable_naming_convention,
        model_exchange=model_exchange,
        co_simulation=co_simulation,
        scheduled_execution=scheduled_execution,
        unit_definitions=unit_definitions,
        type_definitions=type_definitions,
        log_categories=log_categories,
        default_experiment=default_experiment,
        vendor_annotations=vendor_annotations,
        model_variables=model_variables,
        model_structure=model_structure,
    )


def _parse_model_exchange(elem: Element) -> ModelExchange:
    """Parse ModelExchange element"""
    model_identifier = elem.get("modelIdentifier")
    if model_identifier is None:
        raise ValueError("ModelExchange element must have modelIdentifier attribute")
    needs_execution_tool = elem.get("needsExecutionTool")
    can_be_instantiated_only_once_per_process = elem.get(
        "canBeInstantiatedOnlyOncePerProcess"
    )
    can_get_and_set_fmu_state = elem.get("canGetAndSetFMUState")
    can_serialize_fmu_state = elem.get("canSerializeFMUState")
    provides_directional_derivatives = elem.get("providesDirectionalDerivatives")
    provides_adjoint_derivatives = elem.get("providesAdjointDerivatives")
    provides_per_element_dependencies = elem.get("providesPerElementDependencies")
    needs_completed_integrator_step = elem.get("needsCompletedIntegratorStep")
    provides_evaluate_discrete_states = elem.get("providesEvaluateDiscreteStates")

    return ModelExchange(
        model_identifier=model_identifier,
        needs_execution_tool=_str_to_bool(needs_execution_tool),
        can_be_instantiated_only_once_per_process=_str_to_bool(
            can_be_instantiated_only_once_per_process
        ),
        can_get_and_set_fmu_state=_str_to_bool(can_get_and_set_fmu_state),
        can_serialize_fmu_state=_str_to_bool(can_serialize_fmu_state),
        provides_directional_derivatives=_str_to_bool(provides_directional_derivatives),
        provides_adjoint_derivatives=_str_to_bool(provides_adjoint_derivatives),
        provides_per_element_dependencies=_str_to_bool(
            provides_per_element_dependencies
        ),
        needs_completed_integrator_step=_str_to_bool(needs_completed_integrator_step),
        provides_evaluate_discrete_states=_str_to_bool(
            provides_evaluate_discrete_states
        ),
    )


def _parse_co_simulation(elem: Element) -> CoSimulation:
    """Parse CoSimulation element"""
    model_identifier = elem.get("modelIdentifier")
    if model_identifier is None:
        raise ValueError("CoSimulation element must have modelIdentifier attribute")
    needs_execution_tool = elem.get("needsExecutionTool")
    can_be_instantiated_only_once_per_process = elem.get(
        "canBeInstantiatedOnlyOncePerProcess"
    )
    can_get_and_set_fmu_state = elem.get("canGetAndSetFMUState")
    can_serialize_fmu_state = elem.get("canSerializeFMUState")
    provides_directional_derivatives = elem.get("providesDirectionalDerivatives")
    provides_adjoint_derivatives = elem.get("providesAdjointDerivatives")
    provides_per_element_dependencies = elem.get("providesPerElementDependencies")
    can_handle_variable_communication_step_size = elem.get(
        "canHandleVariableCommunicationStepSize"
    )
    fixed_internal_step_size = elem.get("fixedInternalStepSize")
    max_output_derivative_order = elem.get("maxOutputDerivativeOrder")
    recommended_intermediate_input_smoothness = elem.get(
        "recommendedIntermediateInputSmoothness"
    )
    provides_intermediate_update = elem.get("providesIntermediateUpdate")
    might_return_early_from_do_step = elem.get("mightReturnEarlyFromDoStep")
    can_return_early_after_intermediate_update = elem.get(
        "canReturnEarlyAfterIntermediateUpdate"
    )
    has_event_mode = elem.get("hasEventMode")
    provides_evaluate_discrete_states = elem.get("providesEvaluateDiscreteStates")

    if fixed_internal_step_size is not None:
        fixed_internal_step_size = float(fixed_internal_step_size)
    if max_output_derivative_order is not None:
        max_output_derivative_order = int(max_output_derivative_order)
    if recommended_intermediate_input_smoothness is not None:
        recommended_intermediate_input_smoothness = int(
            recommended_intermediate_input_smoothness
        )

    return CoSimulation(
        model_identifier=model_identifier,
        needs_execution_tool=_str_to_bool(needs_execution_tool),
        can_be_instantiated_only_once_per_process=_str_to_bool(
            can_be_instantiated_only_once_per_process
        ),
        can_get_and_set_fmu_state=_str_to_bool(can_get_and_set_fmu_state),
        can_serialize_fmu_state=_str_to_bool(can_serialize_fmu_state),
        provides_directional_derivatives=_str_to_bool(provides_directional_derivatives),
        provides_adjoint_derivatives=_str_to_bool(provides_adjoint_derivatives),
        provides_per_element_dependencies=_str_to_bool(
            provides_per_element_dependencies
        ),
        can_handle_variable_communication_step_size=_str_to_bool(
            can_handle_variable_communication_step_size
        ),
        fixed_internal_step_size=fixed_internal_step_size,
        max_output_derivative_order=max_output_derivative_order,
        recommended_intermediate_input_smoothness=recommended_intermediate_input_smoothness,
        provides_intermediate_update=_str_to_bool(provides_intermediate_update),
        might_return_early_from_do_step=_str_to_bool(might_return_early_from_do_step),
        can_return_early_after_intermediate_update=_str_to_bool(
            can_return_early_after_intermediate_update
        ),
        has_event_mode=_str_to_bool(has_event_mode),
        provides_evaluate_discrete_states=_str_to_bool(
            provides_evaluate_discrete_states
        ),
    )


def _parse_scheduled_execution(elem: Element) -> ScheduledExecution:
    """Parse ScheduledExecution element"""
    model_identifier = elem.get("modelIdentifier")
    if model_identifier is None:
        raise ValueError(
            "ScheduledExecution element must have modelIdentifier attribute"
        )
    needs_execution_tool = elem.get("needsExecutionTool")
    can_be_instantiated_only_once_per_process = elem.get(
        "canBeInstantiatedOnlyOncePerProcess"
    )
    can_get_and_set_fmu_state = elem.get("canGetAndSetFMUState")
    can_serialize_fmu_state = elem.get("canSerializeFMUState")
    provides_directional_derivatives = elem.get("providesDirectionalDerivatives")
    provides_adjoint_derivatives = elem.get("providesAdjointDerivatives")
    provides_per_element_dependencies = elem.get("providesPerElementDependencies")

    return ScheduledExecution(
        model_identifier=model_identifier,
        needs_execution_tool=_str_to_bool(needs_execution_tool),
        can_be_instantiated_only_once_per_process=_str_to_bool(
            can_be_instantiated_only_once_per_process
        ),
        can_get_and_set_fmu_state=_str_to_bool(can_get_and_set_fmu_state),
        can_serialize_fmu_state=_str_to_bool(can_serialize_fmu_state),
        provides_directional_derivatives=_str_to_bool(provides_directional_derivatives),
        provides_adjoint_derivatives=_str_to_bool(provides_adjoint_derivatives),
        provides_per_element_dependencies=_str_to_bool(
            provides_per_element_dependencies
        ),
    )


def _parse_unit_definitions(elem: Element) -> UnitDefinitions:
    """Parse UnitDefinitions element"""
    units = []
    for unit_elem in elem.findall("Unit"):
        unit = _parse_unit(unit_elem)
        units.append(unit)
    return UnitDefinitions(units=units)


def _parse_unit(elem: Element) -> Unit:
    """Parse Unit element"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Unit element `{elem.tag}` must have name attribute")

    base_unit_elem = elem.find("BaseUnit")
    base_unit = None
    if base_unit_elem is not None:
        base_unit = _parse_base_unit(base_unit_elem)

    display_units = []
    for display_unit_elem in elem.findall("DisplayUnit"):
        display_unit = _parse_display_unit(display_unit_elem)
        display_units.append(display_unit)

    return Unit(
        name=name,
        base_unit=base_unit,
        display_units=display_units if display_units else None,
    )


def _parse_base_unit(elem: Element) -> BaseUnit:
    """Parse BaseUnit element"""
    kg = elem.get("kg")
    m = elem.get("m")
    s = elem.get("s")
    a = elem.get("A")
    k = elem.get("K")
    mol = elem.get("mol")
    cd = elem.get("cd")
    rad = elem.get("rad")
    factor = elem.get("factor")
    offset = elem.get("offset")

    return BaseUnit(
        kg=int(kg) if kg is not None else 0,
        m=int(m) if m is not None else 0,
        s=int(s) if s is not None else 0,
        a=int(a) if a is not None else 0,
        k=int(k) if k is not None else 0,
        mol=int(mol) if mol is not None else 0,
        cd=int(cd) if cd is not None else 0,
        rad=int(rad) if rad is not None else 0,
        factor=float(factor) if factor is not None else 1.0,
        offset=float(offset) if offset is not None else 0.0,
    )


def _parse_display_unit(elem: Element) -> DisplayUnit:
    """Parse DisplayUnit element"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"DisplayUnit element `{elem.tag}` must have name attribute")
    factor = elem.get("factor")
    offset = elem.get("offset")

    return DisplayUnit(
        name=name,
        factor=float(factor) if factor is not None else 1.0,
        offset=float(offset) if offset is not None else 0.0,
    )


def _parse_type_definitions(elem: Element) -> TypeDefinitions:
    """Parse TypeDefinitions element

    In FMI 3.0, type definitions are directly under TypeDefinitions as:
    <Float32Type>, <Float64Type>, <Int8Type>, etc.
    """
    types = []

    # FMI 3.0 uses direct type elements like <Float64Type name="...">
    for child in elem:
        tag = child.tag
        name = child.get("name")
        if name is None:
            continue
        description = child.get("description")

        float32 = None
        float64 = None
        int8 = None
        int16 = None
        int32 = None
        int64 = None
        uint8 = None
        uint16 = None
        uint32 = None
        uint64 = None
        boolean = None
        string = None
        binary = None
        enumeration = None
        clock = None

        if tag == "Float32Type":
            float32 = _parse_float32_type(child)
        elif tag == "Float64Type":
            float64 = _parse_float64_type(child)
        elif tag == "Int8Type":
            int8 = _parse_int8_type(child)
        elif tag == "Int16Type":
            int16 = _parse_int16_type(child)
        elif tag == "Int32Type":
            int32 = _parse_int32_type(child)
        elif tag == "Int64Type":
            int64 = _parse_int64_type(child)
        elif tag == "UInt8Type":
            uint8 = _parse_uint8_type(child)
        elif tag == "UInt16Type":
            uint16 = _parse_uint16_type(child)
        elif tag == "UInt32Type":
            uint32 = _parse_uint32_type(child)
        elif tag == "UInt64Type":
            uint64 = _parse_uint64_type(child)
        elif tag == "BooleanType":
            boolean = BooleanType()
        elif tag == "StringType":
            string = StringType()
        elif tag == "BinaryType":
            binary = _parse_binary_type(child)
        elif tag == "EnumerationType":
            enumeration = _parse_enumeration_type(child)
        elif tag == "ClockType":
            clock = _parse_clock_type(child)
        else:
            continue  # Unknown type, skip

        types.append(
            Type(
                name=name,
                description=description,
                float32=float32,
                float64=float64,
                int8=int8,
                int16=int16,
                int32=int32,
                int64=int64,
                uint8=uint8,
                uint16=uint16,
                uint32=uint32,
                uint64=uint64,
                boolean=boolean,
                string=string,
                binary=binary,
                enumeration=enumeration,
                clock=clock,
            )
        )

    return TypeDefinitions(types=types)


def _parse_type_definition(elem: Element) -> Type:
    """Parse Type element"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Type element '{elem.tag}' must have name attribute")
    description = elem.get("description")

    # Determine which type is present
    float32_elem = elem.find("Float32")
    float64_elem = elem.find("Float64")
    int8_elem = elem.find("Int8")
    int16_elem = elem.find("Int16")
    int32_elem = elem.find("Int32")
    int64_elem = elem.find("Int64")
    uint8_elem = elem.find("UInt8")
    uint16_elem = elem.find("UInt16")
    uint32_elem = elem.find("UInt32")
    uint64_elem = elem.find("UInt64")
    boolean_elem = elem.find("Boolean")
    string_elem = elem.find("String")
    binary_elem = elem.find("Binary")
    enumeration_elem = elem.find("Enumeration")
    clock_elem = elem.find("Clock")

    float32 = None
    float64 = None
    int8 = None
    int16 = None
    int32 = None
    int64 = None
    uint8 = None
    uint16 = None
    uint32 = None
    uint64 = None
    boolean = None
    string = None
    binary = None
    enumeration = None
    clock = None

    if float32_elem is not None:
        float32 = _parse_float32_type(float32_elem)
    elif float64_elem is not None:
        float64 = _parse_float64_type(float64_elem)
    elif int8_elem is not None:
        int8 = _parse_int8_type(int8_elem)
    elif int16_elem is not None:
        int16 = _parse_int16_type(int16_elem)
    elif int32_elem is not None:
        int32 = _parse_int32_type(int32_elem)
    elif int64_elem is not None:
        int64 = _parse_int64_type(int64_elem)
    elif uint8_elem is not None:
        uint8 = _parse_uint8_type(uint8_elem)
    elif uint16_elem is not None:
        uint16 = _parse_uint16_type(uint16_elem)
    elif uint32_elem is not None:
        uint32 = _parse_uint32_type(uint32_elem)
    elif uint64_elem is not None:
        uint64 = _parse_uint64_type(uint64_elem)
    elif boolean_elem is not None:
        boolean = BooleanType()
    elif string_elem is not None:
        string = StringType()
    elif binary_elem is not None:
        binary = _parse_binary_type(binary_elem)
    elif enumeration_elem is not None:
        enumeration = _parse_enumeration_type(enumeration_elem)
    elif clock_elem is not None:
        clock = _parse_clock_type(clock_elem)

    return Type(
        name=name,
        description=description,
        float32=float32,
        float64=float64,
        int8=int8,
        int16=int16,
        int32=int32,
        int64=int64,
        uint8=uint8,
        uint16=uint16,
        uint32=uint32,
        uint64=uint64,
        boolean=boolean,
        string=string,
        binary=binary,
        enumeration=enumeration,
        clock=clock,
    )


def _parse_float32_type(elem: Element) -> Float32Type:
    """Parse Float32 element in Type"""
    quantity = elem.get("quantity")
    unit = elem.get("unit")
    display_unit = elem.get("displayUnit")
    relative_quantity = elem.get("relativeQuantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    nominal = elem.get("nominal")
    unbounded = elem.get("unbounded")

    return Float32Type(
        quantity=quantity,
        unit=unit,
        display_unit=display_unit,
        relative_quantity=_str_to_bool(relative_quantity),
        min_value=float(min_value) if min_value is not None else None,
        max_value=float(max_value) if max_value is not None else None,
        nominal=float(nominal) if nominal is not None else None,
        unbounded=_str_to_bool(unbounded),
    )


def _parse_float64_type(elem: Element) -> Float64Type:
    """Parse Float64 element in Type"""
    quantity = elem.get("quantity")
    unit = elem.get("unit")
    display_unit = elem.get("displayUnit")
    relative_quantity = elem.get("relativeQuantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    nominal = elem.get("nominal")
    unbounded = elem.get("unbounded")

    return Float64Type(
        quantity=quantity,
        unit=unit,
        display_unit=display_unit,
        relative_quantity=_str_to_bool(relative_quantity),
        min_value=float(min_value) if min_value is not None else None,
        max_value=float(max_value) if max_value is not None else None,
        nominal=float(nominal) if nominal is not None else None,
        unbounded=_str_to_bool(unbounded),
    )


def _parse_int8_type(elem: Element) -> Int8Type:
    """Parse Int8 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return Int8Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_int16_type(elem: Element) -> Int16Type:
    """Parse Int16 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return Int16Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_int32_type(elem: Element) -> Int32Type:
    """Parse Int32 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return Int32Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_int64_type(elem: Element) -> Int64Type:
    """Parse Int64 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return Int64Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_uint8_type(elem: Element) -> UInt8Type:
    """Parse UInt8 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return UInt8Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_uint16_type(elem: Element) -> UInt16Type:
    """Parse UInt16 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return UInt16Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_uint32_type(elem: Element) -> UInt32Type:
    """Parse UInt32 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return UInt32Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_uint64_type(elem: Element) -> UInt64Type:
    """Parse UInt64 element in Type"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return UInt64Type(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_binary_type(elem: Element) -> BinaryType:
    """Parse Binary element in Type"""
    mime_type = elem.get("mimeType")
    max_size = elem.get("maxSize")

    return BinaryType(
        mime_type=mime_type,
        max_size=int(max_size) if max_size is not None else None,
    )


def _parse_enumeration_type(elem: Element) -> EnumerationType:
    """Parse Enumeration element in Type"""
    quantity = elem.get("quantity")

    items = []
    for item_elem in elem.findall("Item"):
        name = item_elem.get("name")
        if name is None:
            raise ValueError(f"Item element `{item_elem.tag}` must have name attribute")
        value = item_elem.get("value")
        if value is None:
            raise ValueError(
                f"Item element `{item_elem.tag}` must have value attribute"
            )
        description = item_elem.get("description")
        items.append(Item(name=name, value=int(value), description=description))

    return EnumerationType(quantity=quantity, items=items)


def _parse_clock_type(elem: Element) -> ClockType:
    """Parse Clock element in Type"""
    can_be_deactivated = elem.get("canBeDeactivated")
    priority = elem.get("priority")
    synchronised = elem.get("synchronised")
    support_frequency_resolution = elem.get("supportFrequencyResolution")
    support_tentative_steps = elem.get("supportTentativeSteps")
    resolution = elem.get("resolution")
    interval_counter = elem.get("intervalCounter")

    return ClockType(
        can_be_deactivated=_str_to_bool(can_be_deactivated),
        priority=int(priority) if priority is not None else None,
        synchronised=_str_to_bool(synchronised),
        support_frequency_resolution=float(support_frequency_resolution)
        if support_frequency_resolution is not None
        else None,
        support_tentative_steps=_str_to_bool(support_tentative_steps),
        resolution=int(resolution) if resolution is not None else None,
        interval_counter=int(interval_counter)
        if interval_counter is not None
        else None,
    )


def _parse_log_categories(elem: Element) -> LogCategories:
    """Parse LogCategories element"""
    categories = []
    for category_elem in elem.findall("Category"):
        name = category_elem.get("name")
        if name is None:
            raise ValueError(
                f"Category element `{category_elem.tag}` must have name attribute"
            )
        description = category_elem.get("description")
        categories.append(Category(name=name, description=description))
    return LogCategories(categories=categories)


def _parse_default_experiment(elem: Element) -> DefaultExperiment:
    """Parse DefaultExperiment element"""
    start_time = elem.get("startTime")
    stop_time = elem.get("stopTime")
    tolerance = elem.get("tolerance")
    step_size = elem.get("stepSize")

    return DefaultExperiment(
        start_time=float(start_time) if start_time is not None else None,
        stop_time=float(stop_time) if stop_time is not None else None,
        tolerance=float(tolerance) if tolerance is not None else None,
        step_size=float(step_size) if step_size is not None else None,
    )


def _parse_vendor_annotations(elem: Element) -> Annotation:
    """Parse VendorAnnotations element"""
    tools = []
    for tool_elem in elem.findall("Tool"):
        name = tool_elem.get("name")
        if name is None:
            raise ValueError(f"Tool element `{tool_elem.tag}` must have name attribute")
        # For now, we'll just store the name and not parse the complex content
        tools.append(Tool(name=name, content=None))
    return Annotation(tools=tools)


def _parse_dimensions(elem: Element) -> list[Dimension] | None:
    """Parse Dimension child elements on an arrayable variable"""
    dims: list[Dimension] = []
    for dim_elem in elem.findall("Dimension"):
        start_attr = dim_elem.get("start")
        value_ref_attr = dim_elem.get("valueReference")
        dims.append(
            Dimension(
                start=int(start_attr) if start_attr is not None else None,
                value_reference=int(value_ref_attr)
                if value_ref_attr is not None
                else None,
            )
        )
    return dims or None


def _parse_model_variables(elem: Element) -> list[Variable]:
    """Parse ModelVariables element into a flat list"""
    variables = []
    # Find all variable types in the model variables section
    for var_type in [
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Boolean",
        "String",
        "Binary",
        "Enumeration",
        "Clock",
    ]:
        for variable_elem in elem.findall(var_type):
            variable = _parse_variable(variable_elem, var_type)
            variables.append(variable)
    return variables


def _parse_variable(elem: Element, var_type: str) -> Variable:
    """Parse a variable element based on its type"""
    # Create the appropriate variable based on the element tag
    if var_type == "Float32":
        return Variable(float32=_parse_float32_variable(elem))
    elif var_type == "Float64":
        return Variable(float64=_parse_float64_variable(elem))
    elif var_type == "Int8":
        return Variable(int8=_parse_int8_variable(elem))
    elif var_type == "Int16":
        return Variable(int16=_parse_int16_variable(elem))
    elif var_type == "Int32":
        return Variable(int32=_parse_int32_variable(elem))
    elif var_type == "Int64":
        return Variable(int64=_parse_int64_variable(elem))
    elif var_type == "UInt8":
        return Variable(uint8=_parse_uint8_variable(elem))
    elif var_type == "UInt16":
        return Variable(uint16=_parse_uint16_variable(elem))
    elif var_type == "UInt32":
        return Variable(uint32=_parse_uint32_variable(elem))
    elif var_type == "UInt64":
        return Variable(uint64=_parse_uint64_variable(elem))
    elif var_type == "Boolean":
        return Variable(boolean=_parse_boolean_variable(elem))
    elif var_type == "String":
        return Variable(string=_parse_string_variable(elem))
    elif var_type == "Binary":
        return Variable(binary=_parse_binary_variable(elem))
    elif var_type == "Enumeration":
        return Variable(enumeration=_parse_enumeration_variable(elem))
    elif var_type == "Clock":
        return Variable(clock=_parse_clock_variable(elem))
    else:
        raise ValueError(f"Unknown variable type: {var_type}")


def _parse_float32_variable(elem: Element) -> Float32Variable:
    """Parse Float32 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Float32 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Float32 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    unit = elem.get("unit")
    display_unit = elem.get("displayUnit")
    relative_quantity = elem.get("relativeQuantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    nominal = elem.get("nominal")
    unbounded = elem.get("unbounded")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [float(x) for x in start.split()]

    return Float32Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        unit=unit,
        display_unit=display_unit,
        relative_quantity=_str_to_bool(relative_quantity),
        min_value=float(min_value) if min_value is not None else None,
        max_value=float(max_value) if max_value is not None else None,
        nominal=float(nominal) if nominal is not None else None,
        unbounded=_str_to_bool(unbounded),
        dimensions=dimensions,
        start=start_values,
    )


def _parse_float64_variable(elem: Element) -> Float64Variable:
    """Parse Float64 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Float64 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Float64 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    unit = elem.get("unit")
    display_unit = elem.get("displayUnit")
    relative_quantity = elem.get("relativeQuantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    nominal = elem.get("nominal")
    unbounded = elem.get("unbounded")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [float(x) for x in start.split()]

    return Float64Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        unit=unit,
        display_unit=display_unit,
        relative_quantity=_str_to_bool(relative_quantity),
        min_value=float(min_value) if min_value is not None else None,
        max_value=float(max_value) if max_value is not None else None,
        nominal=float(nominal) if nominal is not None else None,
        unbounded=_str_to_bool(unbounded),
        dimensions=dimensions,
        start=start_values,
    )


def _parse_int8_variable(elem: Element) -> Int8Variable:
    """Parse Int8 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Int8 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Int8 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return Int8Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_int16_variable(elem: Element) -> Int16Variable:
    """Parse Int16 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Int16 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Int16 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return Int16Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_int32_variable(elem: Element) -> Int32Variable:
    """Parse Int32 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Int32 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Int32 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return Int32Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_int64_variable(elem: Element) -> Int64Variable:
    """Parse Int64 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Int64 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Int64 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return Int64Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_uint8_variable(elem: Element) -> UInt8Variable:
    """Parse UInt8 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"UInt8 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"UInt8 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return UInt8Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_uint16_variable(elem: Element) -> UInt16Variable:
    """Parse UInt16 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"UInt16 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"UInt16 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return UInt16Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_uint32_variable(elem: Element) -> UInt32Variable:
    """Parse UInt32 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"UInt32 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"UInt32 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return UInt32Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_uint64_variable(elem: Element) -> UInt64Variable:
    """Parse UInt64 element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"UInt64 element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"UInt64 element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return UInt64Variable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_boolean_variable(elem: Element) -> BooleanVariable:
    """Parse Boolean element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Boolean element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Boolean element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [x.lower() == "true" for x in start.split()]

    return BooleanVariable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_string_variable(elem: Element) -> StringVariable:
    """Parse String element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"String element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"String element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = start.split()

    return StringVariable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_binary_variable(elem: Element) -> BinaryVariable:
    """Parse Binary element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Binary element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Binary element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    declared_type = elem.get("declaredType")
    mime_type = elem.get("mimeType")
    max_size = elem.get("maxSize")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [bytes.fromhex(x) for x in start.split()]

    return BinaryVariable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        declared_type=declared_type,
        mime_type=mime_type,
        max_size=int(max_size) if max_size is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_enumeration_variable(elem: Element) -> EnumerationVariable:
    """Parse Enumeration element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Enumeration element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Enumeration element `{elem.tag}` must have valueReference attribute"
        )
    declared_type = elem.get("declaredType")
    if declared_type is None:
        raise ValueError(
            f"Enumeration element `{elem.tag}` must have declaredType attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")
    dimensions = _parse_dimensions(elem)

    start_values = None
    if start is not None:
        start_values = [int(x) for x in start.split()]

    return EnumerationVariable(
        name=name,
        value_reference=int(value_reference),
        declared_type=declared_type,
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        variability=VariabilityEnum(variability)
        if variability
        else VariabilityEnum.continuous,
        initial=InitialEnum(initial) if initial else None,
        can_handle_multiple_set_per_time_instant=_str_to_bool(
            can_handle_multiple_set_per_time_instant
        ),
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        dimensions=dimensions,
        start=start_values,
    )


def _parse_clock_variable(elem: Element) -> ClockVariable:
    """Parse Clock element in ModelVariables"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"Clock element `{elem.tag}` must have name attribute")
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"Clock element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    declared_type = elem.get("declaredType")
    can_be_deactivated = elem.get("canBeDeactivated")
    priority = elem.get("priority")
    synchronised = elem.get("synchronised")
    support_frequency_resolution = elem.get("supportFrequencyResolution")
    support_tentative_steps = elem.get("supportTentativeSteps")
    resolution = elem.get("resolution")
    interval_counter = elem.get("intervalCounter")

    return ClockVariable(
        name=name,
        value_reference=int(value_reference),
        description=description,
        causality=CausalityEnum(causality) if causality else CausalityEnum.local,
        declared_type=declared_type,
        can_be_deactivated=_str_to_bool(can_be_deactivated),
        priority=int(priority) if priority is not None else None,
        synchronised=_str_to_bool(synchronised),
        support_frequency_resolution=float(support_frequency_resolution)
        if support_frequency_resolution is not None
        else None,
        support_tentative_steps=_str_to_bool(support_tentative_steps),
        resolution=int(resolution) if resolution is not None else None,
        interval_counter=int(interval_counter)
        if interval_counter is not None
        else None,
    )


def _parse_model_structure(elem: Element) -> ModelStructure:
    """Parse ModelStructure element"""
    outputs = []
    for output_elem in elem.findall("Output"):
        value_reference = output_elem.get("valueReference")
        if value_reference is None:
            raise ValueError(
                f"Output element `{output_elem.tag}` must have valueReference attribute"
            )
        dependencies = output_elem.get("dependencies")
        dependencies_kind = output_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [x for x in dependencies_kind.split()]

        outputs.append(
            Unknown(
                value_reference=int(value_reference),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    continuous_state_derivatives = []
    for derivative_elem in elem.findall("ContinuousStateDerivative"):
        value_reference = derivative_elem.get("valueReference")
        if value_reference is None:
            raise ValueError(
                f"ContinuousStateDerivative element `{derivative_elem.tag}` must have valueReference attribute"
            )
        dependencies = derivative_elem.get("dependencies")
        dependencies_kind = derivative_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [x for x in dependencies_kind.split()]

        continuous_state_derivatives.append(
            Unknown(
                value_reference=int(value_reference),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    clocked_states = []
    for clocked_elem in elem.findall("ClockedState"):
        value_reference = clocked_elem.get("valueReference")
        if value_reference is None:
            raise ValueError(
                f"ClockedState element `{clocked_elem.tag}` must have valueReference attribute"
            )
        dependencies = clocked_elem.get("dependencies")
        dependencies_kind = clocked_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [x for x in dependencies_kind.split()]

        clocked_states.append(
            Unknown(
                value_reference=int(value_reference),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    initial_unknowns = []
    for initial_elem in elem.findall("InitialUnknown"):
        value_reference = initial_elem.get("valueReference")
        if value_reference is None:
            raise ValueError(
                f"InitialUnknown element `{initial_elem.tag}` must have valueReference attribute"
            )
        dependencies = initial_elem.get("dependencies")
        dependencies_kind = initial_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [x for x in dependencies_kind.split()]

        initial_unknowns.append(
            Unknown(
                value_reference=int(value_reference),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    event_indicators = []
    for event_elem in elem.findall("EventIndicator"):
        value_reference = event_elem.get("valueReference")
        if value_reference is None:
            raise ValueError(
                f"EventIndicator element `{event_elem.tag}` must have valueReference attribute"
            )
        dependencies = event_elem.get("dependencies")
        dependencies_kind = event_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [x for x in dependencies_kind.split()]

        event_indicators.append(
            Unknown(
                value_reference=int(value_reference),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    return ModelStructure(
        outputs=outputs if outputs else None,
        continuous_state_derivatives=continuous_state_derivatives
        if continuous_state_derivatives
        else None,
        clocked_states=clocked_states if clocked_states else None,
        initial_unknowns=initial_unknowns if initial_unknowns else None,
        event_indicators=event_indicators if event_indicators else None,
    )


def _str_to_bool(value: str | None) -> bool | None:
    """Convert string to boolean, handling common XML boolean representations"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in ("true", "1", "yes", "on"):
        return True
    elif value_lower in ("false", "0", "no", "off", ""):
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to boolean")
