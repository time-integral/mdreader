import pathlib
import xml.etree.ElementTree as ET
import zipfile
from enum import Enum
from typing import Annotated
from xml.etree.ElementTree import Element

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "read_model_description",
    "CausalityEnum",
    "VariabilityEnum",
    "InitialEnum",
    "VariableNamingConventionEnum",
    "DependenciesKindEnum",
    "BaseUnit",
    "DisplayUnit",
    "Unit",
    "Item",
    "RealSimpleType",
    "IntegerSimpleType",
    "BooleanSimpleType",
    "StringSimpleType",
    "EnumerationSimpleType",
    "SimpleType",
    "File",
    "SourceFiles",
    "ModelExchange",
    "CoSimulation",
    "Category",
    "LogCategories",
    "DefaultExperiment",
    "Tool",
    "Annotation",
    "UnknownDependency",
    "VariableDependency",
    "InitialUnknown",
    "InitialUnknowns",
    "ModelStructure",
    "RealVariable",
    "IntegerVariable",
    "BooleanVariable",
    "StringVariable",
    "EnumerationVariable",
    "ScalarVariable",
    "UnitDefinitions",
    "TypeDefinitions",
    "FmiModelDescription",
]


def read_model_description(filename: str | pathlib.Path) -> "FmiModelDescription":
    """Read and parse an FMI 2.0 model description XML file

    Args:
        filename (str | pathlib.Path): Path to the FMI 2.0 model description XML file or FMU directory or FMU file

    Returns:
        FmiModelDescription: Parsed FMI 2.0 model description
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


class CausalityEnum(str, Enum):
    parameter = "parameter"
    calculatedParameter = "calculatedParameter"
    input = "input"
    output = "output"
    local = "local"
    independent = "independent"


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


class DependenciesKindEnum(str, Enum):
    dependent = "dependent"
    constant = "constant"
    fixed = "fixed"
    tunable = "tunable"
    discrete = "discrete"


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


class RealSimpleType(BaseModel):
    """Real simple type definition"""

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
                    f"RealSimpleType: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert RealSimpleType to XML Element"""
        element = Element("Real")
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


class IntegerSimpleType(BaseModel):
    """Integer simple type definition"""

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
                    f"IntegerSimpleType: max ({self.max_value}) must be >= min ({self.min_value})"
                )
        return self

    def to_xml(self) -> Element:
        """Convert IntegerSimpleType to XML Element"""
        element = Element("Integer")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        return element


class BooleanSimpleType(BaseModel):
    """Boolean simple type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    pass  # Boolean types have no specific attributes

    def to_xml(self) -> Element:
        """Convert BooleanSimpleType to XML Element"""
        element = Element("Boolean")
        return element


class StringSimpleType(BaseModel):
    """String simple type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    pass  # String types have no specific attributes

    def to_xml(self) -> Element:
        """Convert StringSimpleType to XML Element"""
        element = Element("String")
        return element


class EnumerationSimpleType(BaseModel):
    """Enumeration simple type definition"""

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
        """Convert EnumerationSimpleType to XML Element"""
        element = Element("Enumeration")
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.items is not None:
            for item in self.items:
                element.append(item.to_xml())
        return element


class SimpleType(BaseModel):
    """Simple type definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    name: Annotated[
        str,
        Field(
            ...,
            alias="name",
            description='Name of SimpleType element. "name" must be unique with respect to all other elements of the TypeDefinitions list. Furthermore, "name" of a SimpleType must be different to all "name"s of ScalarVariable.',
        ),
    ]
    description: Annotated[
        str | None,
        Field(alias="description", description="Description of the SimpleType"),
    ] = None

    # Type-specific definitions (only one should be present)
    real: Annotated[
        RealSimpleType | None,
        Field(alias="Real", description="Real simple type definition"),
    ] = None
    integer: Annotated[
        IntegerSimpleType | None,
        Field(alias="Integer", description="Integer simple type definition"),
    ] = None
    boolean: Annotated[
        BooleanSimpleType | None,
        Field(alias="Boolean", description="Boolean simple type definition"),
    ] = None
    string: Annotated[
        StringSimpleType | None,
        Field(alias="String", description="String simple type definition"),
    ] = None
    enumeration: Annotated[
        EnumerationSimpleType | None,
        Field(alias="Enumeration", description="Enumeration simple type definition"),
    ] = None

    def get_type_category(self):
        """Get the type category based on which field is set"""
        if self.real is not None:
            return "Real"
        elif self.integer is not None:
            return "Integer"
        elif self.boolean is not None:
            return "Boolean"
        elif self.string is not None:
            return "String"
        elif self.enumeration is not None:
            return "Enumeration"
        return None

    def to_xml(self) -> Element:
        """Convert SimpleType to XML Element"""
        element = Element("SimpleType")
        element.set("name", self.name)
        if self.description is not None:
            element.set("description", self.description)

        # Add the appropriate type element
        if self.real is not None:
            element.append(self.real.to_xml())
        elif self.integer is not None:
            element.append(self.integer.to_xml())
        elif self.boolean is not None:
            element.append(self.boolean.to_xml())
        elif self.string is not None:
            element.append(self.string.to_xml())
        elif self.enumeration is not None:
            element.append(self.enumeration.to_xml())

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
    """Model Exchange interface definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    model_identifier: Annotated[str, Field(..., alias="modelIdentifier")]
    needs_execution_tool: Annotated[
        bool | None, Field(default=False, alias="needsExecutionTool")
    ] = False
    completed_integrator_step_not_needed: Annotated[
        bool | None, Field(default=False, alias="completedIntegratorStepNotNeeded")
    ] = False
    can_be_instantiated_only_once_per_process: Annotated[
        bool | None,
        Field(default=False, alias="canBeInstantiatedOnlyOncePerProcess"),
    ] = False
    can_not_use_memory_management_functions: Annotated[
        bool | None, Field(default=False, alias="canNotUseMemoryManagementFunctions")
    ] = False
    can_get_and_set_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canGetAndSetFMUstate")
    ] = False
    can_serialize_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canSerializeFMUstate")
    ] = False
    provides_directional_derivative: Annotated[
        bool | None, Field(default=False, alias="providesDirectionalDerivative")
    ] = False
    source_files: Annotated[
        SourceFiles | None, Field(default=None, alias="SourceFiles")
    ] = None

    def to_xml(self) -> Element:
        """Convert ModelExchange to XML Element"""
        element = Element("ModelExchange")
        element.set("modelIdentifier", self.model_identifier)
        if self.needs_execution_tool is not None and self.needs_execution_tool:
            element.set("needsExecutionTool", str(self.needs_execution_tool).lower())
        if (
            self.completed_integrator_step_not_needed is not None
            and self.completed_integrator_step_not_needed
        ):
            element.set(
                "completedIntegratorStepNotNeeded",
                str(self.completed_integrator_step_not_needed).lower(),
            )
        if (
            self.can_be_instantiated_only_once_per_process is not None
            and self.can_be_instantiated_only_once_per_process
        ):
            element.set(
                "canBeInstantiatedOnlyOncePerProcess",
                str(self.can_be_instantiated_only_once_per_process).lower(),
            )
        if (
            self.can_not_use_memory_management_functions is not None
            and self.can_not_use_memory_management_functions
        ):
            element.set(
                "canNotUseMemoryManagementFunctions",
                str(self.can_not_use_memory_management_functions).lower(),
            )
        if (
            self.can_get_and_set_fmu_state is not None
            and self.can_get_and_set_fmu_state
        ):
            element.set(
                "canGetAndSetFMUstate", str(self.can_get_and_set_fmu_state).lower()
            )
        if self.can_serialize_fmu_state is not None and self.can_serialize_fmu_state:
            element.set(
                "canSerializeFMUstate", str(self.can_serialize_fmu_state).lower()
            )
        if (
            self.provides_directional_derivative is not None
            and self.provides_directional_derivative
        ):
            element.set(
                "providesDirectionalDerivative",
                str(self.provides_directional_derivative).lower(),
            )
        if self.source_files is not None:
            element.append(self.source_files.to_xml())
        return element


class CoSimulation(BaseModel):
    """Co-Simulation interface definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    model_identifier: Annotated[str, Field(..., alias="modelIdentifier")]
    needs_execution_tool: Annotated[
        bool | None, Field(default=False, alias="needsExecutionTool")
    ] = False
    can_handle_variable_communication_step_size: Annotated[
        bool | None,
        Field(default=False, alias="canHandleVariableCommunicationStepSize"),
    ] = False
    can_interpolate_inputs: Annotated[
        bool | None, Field(default=False, alias="canInterpolateInputs")
    ] = False
    max_output_derivative_order: Annotated[
        int | None, Field(default=0, alias="maxOutputDerivativeOrder")
    ] = 0
    can_run_asynchronuously: Annotated[
        bool | None, Field(default=False, alias="canRunAsynchronuously")
    ] = False
    can_be_instantiated_only_once_per_process: Annotated[
        bool | None,
        Field(default=False, alias="canBeInstantiatedOnlyOncePerProcess"),
    ] = False
    can_not_use_memory_management_functions: Annotated[
        bool | None, Field(default=False, alias="canNotUseMemoryManagementFunctions")
    ] = False
    can_get_and_set_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canGetAndSetFMUstate")
    ] = False
    can_serialize_fmu_state: Annotated[
        bool | None, Field(default=False, alias="canSerializeFMUstate")
    ] = False
    provides_directional_derivative: Annotated[
        bool | None, Field(default=False, alias="providesDirectionalDerivative")
    ] = False
    source_files: Annotated[
        SourceFiles | None, Field(default=None, alias="SourceFiles")
    ] = None

    def to_xml(self) -> Element:
        """Convert CoSimulation to XML Element"""
        element = Element("CoSimulation")
        element.set("modelIdentifier", self.model_identifier)
        if self.needs_execution_tool is not None and self.needs_execution_tool:
            element.set("needsExecutionTool", str(self.needs_execution_tool).lower())
        if (
            self.can_handle_variable_communication_step_size is not None
            and self.can_handle_variable_communication_step_size
        ):
            element.set(
                "canHandleVariableCommunicationStepSize",
                str(self.can_handle_variable_communication_step_size).lower(),
            )
        if self.can_interpolate_inputs is not None and self.can_interpolate_inputs:
            element.set(
                "canInterpolateInputs", str(self.can_interpolate_inputs).lower()
            )
        if (
            self.max_output_derivative_order is not None
            and self.max_output_derivative_order != 0
        ):
            element.set(
                "maxOutputDerivativeOrder", str(self.max_output_derivative_order)
            )
        if self.can_run_asynchronuously is not None and self.can_run_asynchronuously:
            element.set(
                "canRunAsynchronuously", str(self.can_run_asynchronuously).lower()
            )
        if (
            self.can_be_instantiated_only_once_per_process is not None
            and self.can_be_instantiated_only_once_per_process
        ):
            element.set(
                "canBeInstantiatedOnlyOncePerProcess",
                str(self.can_be_instantiated_only_once_per_process).lower(),
            )
        if (
            self.can_not_use_memory_management_functions is not None
            and self.can_not_use_memory_management_functions
        ):
            element.set(
                "canNotUseMemoryManagementFunctions",
                str(self.can_not_use_memory_management_functions).lower(),
            )
        if (
            self.can_get_and_set_fmu_state is not None
            and self.can_get_and_set_fmu_state
        ):
            element.set(
                "canGetAndSetFMUstate", str(self.can_get_and_set_fmu_state).lower()
            )
        if self.can_serialize_fmu_state is not None and self.can_serialize_fmu_state:
            element.set(
                "canSerializeFMUstate", str(self.can_serialize_fmu_state).lower()
            )
        if (
            self.provides_directional_derivative is not None
            and self.provides_directional_derivative
        ):
            element.set(
                "providesDirectionalDerivative",
                str(self.provides_directional_derivative).lower(),
            )
        if self.source_files is not None:
            element.append(self.source_files.to_xml())
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

    tools: Annotated[list[Tool], Field(..., alias="Tool")]

    def to_xml(self) -> Element:
        """Convert Annotation to XML Element"""
        element = Element("Annotation")
        if self.tools is not None:
            for tool in self.tools:
                element.append(tool.to_xml())
        return element


class UnknownDependency(BaseModel):
    """Dependency definition for unknown variables"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    index: Annotated[int, Field(..., alias="index")]
    dependencies: Annotated[
        list[int] | None, Field(default=None, alias="dependencies")
    ] = None
    dependencies_kind: Annotated[
        list[DependenciesKindEnum] | None,
        Field(default=None, alias="dependenciesKind"),
    ] = None

    def to_xml(self) -> Element:
        """Convert UnknownDependency to XML Element"""
        element = Element("Unknown")
        element.set("index", str(self.index))
        if self.dependencies is not None:
            element.set("dependencies", " ".join(map(str, self.dependencies)))
        if self.dependencies_kind is not None:
            element.set(
                "dependenciesKind",
                " ".join([kind.value for kind in self.dependencies_kind]),
            )
        return element


class VariableDependency(BaseModel):
    """Variable dependency definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    unknowns: Annotated[list[UnknownDependency], Field(..., alias="Unknown")]

    def to_xml(self) -> Element:
        """Convert VariableDependency to XML Element"""
        element = Element("VariableDependency")
        if self.unknowns is not None:
            for unknown in self.unknowns:
                element.append(unknown.to_xml())
        return element


class InitialUnknown(BaseModel):
    """Initial unknown variable"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    index: Annotated[int, Field(..., alias="index")]
    dependencies: Annotated[
        list[int] | None, Field(default=None, alias="dependencies")
    ] = None
    dependencies_kind: Annotated[
        list[DependenciesKindEnum] | None,
        Field(default=None, alias="dependenciesKind"),
    ] = None

    def to_xml(self) -> Element:
        """Convert InitialUnknown to XML Element"""
        element = Element("Unknown")
        element.set("index", str(self.index))
        if self.dependencies is not None:
            element.set("dependencies", " ".join(map(str, self.dependencies)))
        if self.dependencies_kind is not None:
            element.set(
                "dependenciesKind",
                " ".join([kind.value for kind in self.dependencies_kind]),
            )
        return element


class InitialUnknowns(BaseModel):
    """List of initial unknown variables"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    unknowns: Annotated[list[InitialUnknown], Field(..., alias="Unknown")]

    def to_xml(self) -> Element:
        """Convert InitialUnknowns to XML Element"""
        element = Element("InitialUnknowns")
        if self.unknowns is not None:
            for unknown in self.unknowns:
                element.append(unknown.to_xml())
        return element


class ModelStructure(BaseModel):
    """Model structure definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    outputs: Annotated[
        VariableDependency | None, Field(default=None, alias="Outputs")
    ] = None
    derivatives: Annotated[
        VariableDependency | None, Field(default=None, alias="Derivatives")
    ] = None
    initial_unknowns: Annotated[
        InitialUnknowns | None, Field(default=None, alias="InitialUnknowns")
    ] = None

    def to_xml(self) -> Element:
        """Convert ModelStructure to XML Element"""
        element = Element("ModelStructure")
        if self.outputs is not None:
            outputs_element = self.outputs.to_xml()
            outputs_element.tag = (
                "Outputs"  # Change tag from VariableDependency to Outputs
            )
            element.append(outputs_element)
        if self.derivatives is not None:
            derivatives_element = self.derivatives.to_xml()
            derivatives_element.tag = (
                "Derivatives"  # Change tag from VariableDependency to Derivatives
            )
            element.append(derivatives_element)
        if self.initial_unknowns is not None:
            element.append(self.initial_unknowns.to_xml())
        return element


class RealVariable(BaseModel):
    """Real variable definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / SimpleType providing defaults.",
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
    start: Annotated[
        float | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None
    derivative: Annotated[
        int | None,
        Field(
            default=None,
            alias="derivative",
            description='If present, this variable is the derivative of variable with ScalarVariable index "derivative".',
        ),
    ] = None
    reinit: Annotated[
        bool | None,
        Field(
            default=False,
            alias="reinit",
            description="Only for ModelExchange and if variable is a continuous-time state: If true, state can be reinitialized at an event by the FMU; If false, state will never be reinitialized at an event by the FMU",
        ),
    ] = False

    def to_xml(self) -> Element:
        """Convert RealVariable to XML Element"""
        element = Element("Real")
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
        if self.start is not None:
            element.set("start", str(self.start))
        if self.derivative is not None:
            element.set("derivative", str(self.derivative))
        if self.reinit is not None and self.reinit:
            element.set("reinit", str(self.reinit).lower())
        return element


class IntegerVariable(BaseModel):
    """Integer variable definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / SimpleType providing defaults.",
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
    start: Annotated[
        int | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert IntegerVariable to XML Element"""
        element = Element("Integer")
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.start is not None:
            element.set("start", str(self.start))
        return element


class BooleanVariable(BaseModel):
    """Boolean variable definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / SimpleType providing defaults.",
        ),
    ] = None
    start: Annotated[
        bool | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert BooleanVariable to XML Element"""
        element = Element("Boolean")
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.start is not None:
            element.set("start", str(self.start).lower())
        return element


class StringVariable(BaseModel):
    """String variable definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    declared_type: Annotated[
        str | None,
        Field(
            default=None,
            alias="declaredType",
            description="If present, name of type defined with TypeDefinitions / SimpleType providing defaults.",
        ),
    ] = None
    start: Annotated[
        str | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert StringVariable to XML Element"""
        element = Element("String")
        if self.declared_type is not None:
            element.set("declaredType", self.declared_type)
        if self.start is not None:
            element.set("start", self.start)
        return element


class EnumerationVariable(BaseModel):
    """Enumeration variable definition"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    declared_type: Annotated[
        str,
        Field(
            ...,
            alias="declaredType",
            description="Name of type defined with TypeDefinitions / SimpleType",
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
    start: Annotated[
        int | None,
        Field(
            default=None,
            alias="start",
            description="Value before initialization, if initial=exact or approx. max >= start >= min required",
        ),
    ] = None

    def to_xml(self) -> Element:
        """Convert EnumerationVariable to XML Element"""
        element = Element("Enumeration")
        element.set("declaredType", self.declared_type)
        if self.quantity is not None:
            element.set("quantity", self.quantity)
        if self.min_value is not None:
            element.set("min", str(self.min_value))
        if self.max_value is not None:
            element.set("max", str(self.max_value))
        if self.start is not None:
            element.set("start", str(self.start))
        return element


class ScalarVariable(BaseModel):
    """Scalar variable definition"""

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
            description="Identifier for variable value in FMI2 function calls (not necessarily unique with respect to all variables)",
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
            description="parameter: independent parameter; calculatedParameter: calculated parameter; input/output: can be used in connections; local: variable calculated from other variables; independent: independent variable (usually time)",
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
            description='Only for ModelExchange and only for variables with variability = "input": If present with value = false, then only one fmi2SetXXX call is allowed at one super dense time instant. In other words, this input is not allowed to appear in an algebraic loop.',
        ),
    ] = None

    # Variable type (one of these should be present)
    real: Annotated[
        RealVariable | None,
        Field(default=None, alias="Real", description="Real variable definition"),
    ] = None
    integer: Annotated[
        IntegerVariable | None,
        Field(default=None, alias="Integer", description="Integer variable definition"),
    ] = None
    boolean: Annotated[
        BooleanVariable | None,
        Field(default=None, alias="Boolean", description="Boolean variable definition"),
    ] = None
    string: Annotated[
        StringVariable | None,
        Field(default=None, alias="String", description="String variable definition"),
    ] = None
    enumeration: Annotated[
        EnumerationVariable | None,
        Field(
            default=None,
            alias="Enumeration",
            description="Enumeration variable definition",
        ),
    ] = None

    # Annotations
    annotations: Annotated[
        Annotation | None,
        Field(
            default=None,
            alias="Annotations",
            description="Additional data of the scalar variable, e.g., for the dialog menu or the graphical layout",
        ),
    ] = None

    def get_variable_type(self):
        """Get the type of variable based on which field is set"""
        if self.real is not None:
            return "Real"
        elif self.integer is not None:
            return "Integer"
        elif self.boolean is not None:
            return "Boolean"
        elif self.string is not None:
            return "String"
        elif self.enumeration is not None:
            return "Enumeration"
        return None

    def to_xml(self) -> Element:
        """Convert ScalarVariable to XML Element"""
        element = Element("ScalarVariable")
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

        # Add the appropriate variable type element
        if self.real is not None:
            element.append(self.real.to_xml())
        elif self.integer is not None:
            element.append(self.integer.to_xml())
        elif self.boolean is not None:
            element.append(self.boolean.to_xml())
        elif self.string is not None:
            element.append(self.string.to_xml())
        elif self.enumeration is not None:
            element.append(self.enumeration.to_xml())

        # Add annotations if present
        if self.annotations is not None:
            element.append(self.annotations.to_xml())

        return element


class UnitDefinitions(BaseModel):
    """Unit definitions list"""

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
    """Type definitions list"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    simple_types: Annotated[list[SimpleType], Field(..., alias="SimpleType")]

    def to_xml(self) -> Element:
        """Convert TypeDefinitions to XML Element"""
        element = Element("TypeDefinitions")
        if self.simple_types is not None:
            for simple_type in self.simple_types:
                element.append(simple_type.to_xml())
        return element


class FmiModelDescription(BaseModel):
    """Main FMI model description"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    fmi_version: Annotated[
        str,
        Field(
            ...,
            alias="fmiVersion",
            description='Version of FMI (Clarification for FMI 2.0.2: for FMI 2.0.x revisions fmiVersion is defined as "2.0")',
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
    guid: Annotated[
        str,
        Field(
            ...,
            alias="guid",
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
    number_of_event_indicators: Annotated[
        int | None,
        Field(
            default=None,
            alias="numberOfEventIndicators",
            description="Number of event indicators in the model",
        ),
    ] = None

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
        list[ScalarVariable],
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
        element.set("guid", self.guid)
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
        if self.number_of_event_indicators is not None:
            element.set("numberOfEventIndicators", str(self.number_of_event_indicators))

        # Add optional components
        if self.model_exchange is not None:
            element.append(self.model_exchange.to_xml())
        if self.co_simulation is not None:
            element.append(self.co_simulation.to_xml())
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
    guid = root.get("guid")
    if guid is None:
        raise ValueError("GUID attribute is required")
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
    number_of_event_indicators = root.get("numberOfEventIndicators")
    if number_of_event_indicators is not None:
        number_of_event_indicators = int(number_of_event_indicators)

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
        guid=guid,
        description=description,
        author=author,
        version=version,
        copyright=copyright,
        license=license,
        generation_tool=generation_tool,
        generation_date_and_time=generation_date_and_time,
        variable_naming_convention=variable_naming_convention,
        number_of_event_indicators=number_of_event_indicators,
        model_exchange=model_exchange,
        co_simulation=co_simulation,
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
    completed_integrator_step_not_needed = elem.get("completedIntegratorStepNotNeeded")
    can_be_instantiated_only_once_per_process = elem.get(
        "canBeInstantiatedOnlyOncePerProcess"
    )
    can_not_use_memory_management_functions = elem.get(
        "canNotUseMemoryManagementFunctions"
    )
    can_get_and_set_fmu_state = elem.get("canGetAndSetFMUstate")
    can_serialize_fmu_state = elem.get("canSerializeFMUstate")
    provides_directional_derivative = elem.get("providesDirectionalDerivative")

    source_files_elem = elem.find("SourceFiles")
    source_files = None
    if source_files_elem is not None:
        source_files = _parse_source_files(source_files_elem)

    return ModelExchange(
        model_identifier=model_identifier,
        needs_execution_tool=_str_to_bool(needs_execution_tool),
        completed_integrator_step_not_needed=_str_to_bool(
            completed_integrator_step_not_needed
        ),
        can_be_instantiated_only_once_per_process=_str_to_bool(
            can_be_instantiated_only_once_per_process
        ),
        can_not_use_memory_management_functions=_str_to_bool(
            can_not_use_memory_management_functions
        ),
        can_get_and_set_fmu_state=_str_to_bool(can_get_and_set_fmu_state),
        can_serialize_fmu_state=_str_to_bool(can_serialize_fmu_state),
        provides_directional_derivative=_str_to_bool(provides_directional_derivative),
        source_files=source_files,
    )


def _parse_co_simulation(elem: Element) -> CoSimulation:
    """Parse CoSimulation element"""
    model_identifier = elem.get("modelIdentifier")
    if model_identifier is None:
        raise ValueError("CoSimulation element must have modelIdentifier attribute")
    needs_execution_tool = elem.get("needsExecutionTool")
    can_handle_variable_communication_step_size = elem.get(
        "canHandleVariableCommunicationStepSize"
    )
    can_interpolate_inputs = elem.get("canInterpolateInputs")
    max_output_derivative_order = elem.get("maxOutputDerivativeOrder")
    can_run_asynchronuously = elem.get("canRunAsynchronuously")
    can_be_instantiated_only_once_per_process = elem.get(
        "canBeInstantiatedOnlyOncePerProcess"
    )
    can_not_use_memory_management_functions = elem.get(
        "canNotUseMemoryManagementFunctions"
    )
    can_get_and_set_fmu_state = elem.get("canGetAndSetFMUstate")
    can_serialize_fmu_state = elem.get("canSerializeFMUstate")
    provides_directional_derivative = elem.get("providesDirectionalDerivative")

    if max_output_derivative_order is not None:
        max_output_derivative_order = int(max_output_derivative_order)

    source_files_elem = elem.find("SourceFiles")
    source_files = None
    if source_files_elem is not None:
        source_files = _parse_source_files(source_files_elem)

    return CoSimulation(
        model_identifier=model_identifier,
        needs_execution_tool=_str_to_bool(needs_execution_tool),
        can_handle_variable_communication_step_size=_str_to_bool(
            can_handle_variable_communication_step_size
        ),
        can_interpolate_inputs=_str_to_bool(can_interpolate_inputs),
        max_output_derivative_order=max_output_derivative_order,
        can_run_asynchronuously=_str_to_bool(can_run_asynchronuously),
        can_be_instantiated_only_once_per_process=_str_to_bool(
            can_be_instantiated_only_once_per_process
        ),
        can_not_use_memory_management_functions=_str_to_bool(
            can_not_use_memory_management_functions
        ),
        can_get_and_set_fmu_state=_str_to_bool(can_get_and_set_fmu_state),
        can_serialize_fmu_state=_str_to_bool(can_serialize_fmu_state),
        provides_directional_derivative=_str_to_bool(provides_directional_derivative),
        source_files=source_files,
    )


def _parse_source_files(elem: Element) -> SourceFiles:
    """Parse SourceFiles element"""
    files = []
    for file_elem in elem.findall("File"):
        name = file_elem.get("name")
        if name is None:
            raise ValueError("File element must have name attribute")
        files.append(File(name=name))

    return SourceFiles(files=files)


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
    """Parse TypeDefinitions element"""
    simple_types = []
    for simple_type_elem in elem.findall("SimpleType"):
        simple_type = _parse_simple_type(simple_type_elem)
        simple_types.append(simple_type)
    return TypeDefinitions(simple_types=simple_types)


def _parse_simple_type(elem: Element) -> SimpleType:
    """Parse SimpleType element"""
    name = elem.get("name")
    if name is None:
        raise ValueError(f"SimpleType element '{elem.tag}' must have name attribute")
    description = elem.get("description")

    # Determine which type is present
    real_elem = elem.find("Real")
    integer_elem = elem.find("Integer")
    boolean_elem = elem.find("Boolean")
    string_elem = elem.find("String")
    enumeration_elem = elem.find("Enumeration")

    real = None
    integer = None
    boolean = None
    string = None
    enumeration = None

    if real_elem is not None:
        real = _parse_real_simple_type(real_elem)
    elif integer_elem is not None:
        integer = _parse_integer_simple_type(integer_elem)
    elif boolean_elem is not None:
        boolean = BooleanSimpleType()
    elif string_elem is not None:
        string = StringSimpleType()
    elif enumeration_elem is not None:
        enumeration = _parse_enumeration_simple_type(enumeration_elem)

    return SimpleType(
        name=name,
        description=description,
        real=real,
        integer=integer,
        boolean=boolean,
        string=string,
        enumeration=enumeration,
    )


def _parse_real_simple_type(elem: Element) -> RealSimpleType:
    """Parse Real element in SimpleType"""
    quantity = elem.get("quantity")
    unit = elem.get("unit")
    display_unit = elem.get("displayUnit")
    relative_quantity = elem.get("relativeQuantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    nominal = elem.get("nominal")
    unbounded = elem.get("unbounded")

    return RealSimpleType(
        quantity=quantity,
        unit=unit,
        display_unit=display_unit,
        relative_quantity=_str_to_bool(relative_quantity),
        min_value=float(min_value) if min_value is not None else None,
        max_value=float(max_value) if max_value is not None else None,
        nominal=float(nominal) if nominal is not None else None,
        unbounded=_str_to_bool(unbounded),
    )


def _parse_integer_simple_type(elem: Element) -> IntegerSimpleType:
    """Parse Integer element in SimpleType"""
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")

    return IntegerSimpleType(
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
    )


def _parse_enumeration_simple_type(elem: Element) -> EnumerationSimpleType:
    """Parse Enumeration element in SimpleType"""
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

    return EnumerationSimpleType(quantity=quantity, items=items)


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


def _parse_model_variables(elem: Element) -> list[ScalarVariable]:
    """Parse ModelVariables element into a flat list"""
    variables = []
    for variable_elem in elem.findall("ScalarVariable"):
        variable = _parse_scalar_variable(variable_elem)
        variables.append(variable)
    return variables


def _parse_scalar_variable(elem: Element) -> ScalarVariable:
    """Parse ScalarVariable element"""
    name = elem.get("name")
    if name is None:
        raise ValueError(
            f"ScalarVariable element `{elem.tag}` must have name attribute"
        )
    value_reference = elem.get("valueReference")
    if value_reference is None:
        raise ValueError(
            f"ScalarVariable element `{elem.tag}` must have valueReference attribute"
        )
    description = elem.get("description")
    causality = elem.get("causality")
    variability = elem.get("variability")
    initial = elem.get("initial")
    can_handle_multiple_set_per_time_instant = elem.get(
        "canHandleMultipleSetPerTimeInstant"
    )

    # Determine variable type
    real_elem = elem.find("Real")
    integer_elem = elem.find("Integer")
    boolean_elem = elem.find("Boolean")
    string_elem = elem.find("String")
    enumeration_elem = elem.find("Enumeration")

    real = None
    integer = None
    boolean = None
    string = None
    enumeration = None

    if real_elem is not None:
        real = _parse_real_variable(real_elem)
    elif integer_elem is not None:
        integer = _parse_integer_variable(integer_elem)
    elif boolean_elem is not None:
        boolean = _parse_boolean_variable(boolean_elem)
    elif string_elem is not None:
        string = _parse_string_variable(string_elem)
    elif enumeration_elem is not None:
        enumeration = _parse_enumeration_variable(enumeration_elem)

    # Parse annotations if present
    annotations_elem = elem.find("Annotations")
    annotations = None
    if annotations_elem is not None:
        annotations = _parse_vendor_annotations(annotations_elem)

    return ScalarVariable(
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
        real=real,
        integer=integer,
        boolean=boolean,
        string=string,
        enumeration=enumeration,
        annotations=annotations,
    )


def _parse_real_variable(elem: Element) -> RealVariable:
    """Parse Real element in ScalarVariable"""
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
    derivative = elem.get("derivative")
    reinit = elem.get("reinit")

    return RealVariable(
        declared_type=declared_type,
        quantity=quantity,
        unit=unit,
        display_unit=display_unit,
        relative_quantity=_str_to_bool(relative_quantity),
        min_value=float(min_value) if min_value is not None else None,
        max_value=float(max_value) if max_value is not None else None,
        nominal=float(nominal) if nominal is not None else None,
        unbounded=_str_to_bool(unbounded),
        start=float(start) if start is not None else None,
        derivative=int(derivative) if derivative is not None else None,
        reinit=_str_to_bool(reinit),
    )


def _parse_integer_variable(elem: Element) -> IntegerVariable:
    """Parse Integer element in ScalarVariable"""
    declared_type = elem.get("declaredType")
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")

    return IntegerVariable(
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        start=int(start) if start is not None else None,
    )


def _parse_boolean_variable(elem: Element) -> BooleanVariable:
    """Parse Boolean element in ScalarVariable"""
    declared_type = elem.get("declaredType")
    start = elem.get("start")

    return BooleanVariable(
        declared_type=declared_type,
        start=_str_to_bool(start) if start is not None else None,
    )


def _parse_string_variable(elem: Element) -> StringVariable:
    """Parse String element in ScalarVariable"""
    declared_type = elem.get("declaredType")
    start = elem.get("start")

    return StringVariable(declared_type=declared_type, start=start)


def _parse_enumeration_variable(elem: Element) -> EnumerationVariable:
    """Parse Enumeration element in ScalarVariable"""
    declared_type = elem.get("declaredType")
    if declared_type is None:
        raise ValueError(
            f"Enumeration element `{elem.tag}` must have declaredType attribute"
        )
    quantity = elem.get("quantity")
    min_value = elem.get("min")
    max_value = elem.get("max")
    start = elem.get("start")

    return EnumerationVariable(
        declared_type=declared_type,
        quantity=quantity,
        min_value=int(min_value) if min_value is not None else None,
        max_value=int(max_value) if max_value is not None else None,
        start=int(start) if start is not None else None,
    )


def _parse_model_structure(elem: Element) -> ModelStructure:
    """Parse ModelStructure element"""
    outputs_elem = elem.find("Outputs")
    derivatives_elem = elem.find("Derivatives")
    initial_unknowns_elem = elem.find("InitialUnknowns")

    outputs = (
        _parse_variable_dependency(outputs_elem) if outputs_elem is not None else None
    )
    derivatives = (
        _parse_variable_dependency(derivatives_elem)
        if derivatives_elem is not None
        else None
    )
    initial_unknowns = (
        _parse_initial_unknowns(initial_unknowns_elem)
        if initial_unknowns_elem is not None
        else None
    )

    return ModelStructure(
        outputs=outputs, derivatives=derivatives, initial_unknowns=initial_unknowns
    )


def _parse_variable_dependency(elem: Element) -> VariableDependency:
    """Parse Outputs or Derivatives element"""
    unknowns = []
    for unknown_elem in elem.findall("Unknown"):
        index = unknown_elem.get("index")
        if index is None:
            raise ValueError(
                f"Unknown element `{unknown_elem.tag}` must have index attribute"
            )
        dependencies = unknown_elem.get("dependencies")
        dependencies_kind = unknown_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [
                DependenciesKindEnum(x) for x in dependencies_kind.split()
            ]

        unknowns.append(
            UnknownDependency(
                index=int(index),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    return VariableDependency(unknowns=unknowns)


def _parse_initial_unknowns(elem: Element) -> InitialUnknowns:
    """Parse InitialUnknowns element"""
    unknowns = []
    for unknown_elem in elem.findall("Unknown"):
        index = unknown_elem.get("index")
        if index is None:
            raise ValueError(
                f"Unknown element `{unknown_elem.tag}` must have index attribute"
            )
        dependencies = unknown_elem.get("dependencies")
        dependencies_kind = unknown_elem.get("dependenciesKind")

        dependencies_list = None
        if dependencies:
            dependencies_list = [int(x) for x in dependencies.split()]

        dependencies_kind_list = None
        if dependencies_kind:
            dependencies_kind_list = [
                DependenciesKindEnum(x) for x in dependencies_kind.split()
            ]

        unknowns.append(
            InitialUnknown(
                index=int(index),
                dependencies=dependencies_list,
                dependencies_kind=dependencies_kind_list,
            )
        )

    return InitialUnknowns(unknowns=unknowns)


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
