import pathlib
import xml.etree.ElementTree as ET
import zipfile
from typing import Literal

import mdreader.fmi2 as fmi2
import mdreader.fmi3 as fmi3


def get_fmi_version(
    filename: str | pathlib.Path,
) -> Literal["2.0", "3.0"]:
    """Determine the FMI version of the model description XML file

    Args:
        filename (str | pathlib.Path): Path to the FMI model description XML file or FMU directory or FMU file

    Returns:
        Literal["2.0", "3.0"]: Parsed FMI model description
    """
    filename = pathlib.Path(filename)
    if filename.suffix == ".xml":
        tree = ET.parse(filename)
        root = tree.getroot()
    elif filename.is_dir():
        xml_file = filename / "modelDescription.xml"
        tree = ET.parse(xml_file)
        root = tree.getroot()
    elif filename.suffix == ".fmu":
        with zipfile.ZipFile(filename, "r") as zf:
            with zf.open("modelDescription.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Must be .xml, .fmu, or directory"
        )

    fmi_version = root.attrib.get("fmiVersion")
    if fmi_version is None:
        raise ValueError("fmiVersion attribute not found in modelDescription.xml")

    if fmi_version.startswith("2."):
        return "2.0"
    elif fmi_version.startswith("3."):
        return "3.0"
    else:
        raise ValueError(f"Unsupported FMI version: {fmi_version}")


def read_model_description(
    filename: str | pathlib.Path,
) -> fmi2.FmiModelDescription | fmi3.FmiModelDescription:
    """Read and parse an FMI 2.0/3.0 model description XML file

    Args:
        filename (str | pathlib.Path): Path to the FMI 2.0 model description XML file or FMU directory or FMU file

    Returns:
        FmiModelDescription: Parsed FMI 2.0 model description
    """
    fmi_version = get_fmi_version(filename)
    if fmi_version == "2.0":
        return fmi2.read_model_description(filename)
    elif fmi_version == "3.0":
        return fmi3.read_model_description(filename)
    else:
        raise ValueError(f"Unsupported FMI version: {fmi_version}")


# def fmi_version()
