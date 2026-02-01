"""Unit tests for the main FMI module that handles both FMI 2.0 and 3.0"""

import pathlib
import tempfile
import zipfile

import httpx
import pytest

from mdreader.fmi import get_fmi_version, read_model_description


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


def test_get_fmi_version_with_fmi2_xml():
    """Test get_fmi_version with FMI 2.0 XML content"""
    # Create a minimal FMI 2.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="2.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function
        result = get_fmi_version(temp_xml_path)
        assert result == "2.0"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_get_fmi_version_with_fmi3_xml():
    """Test get_fmi_version with FMI 3.0 XML content"""
    # Create a minimal FMI 3.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="3.0"
        modelName="TestModel"
        instantiationToken="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function
        result = get_fmi_version(temp_xml_path)
        assert result == "3.0"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_get_fmi_version_with_fmi2_xml_subversion():
    """Test get_fmi_version with FMI 2.x version (like 2.1)"""
    # Create a minimal FMI 2.1 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="2.1"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function - should return "2.0" for any 2.x version
        result = get_fmi_version(temp_xml_path)
        assert result == "2.0"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_get_fmi_version_with_fmi3_xml_subversion():
    """Test get_fmi_version with FMI 3.x version (like 3.1)"""
    # Create a minimal FMI 3.1 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="3.1"
        modelName="TestModel"
        instantiationToken="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function - should return "3.0" for any 3.x version
        result = get_fmi_version(temp_xml_path)
        assert result == "3.0"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_get_fmi_version_with_directory():
    """Test get_fmi_version with a directory containing modelDescription.xml"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create modelDescription.xml inside the directory
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <fmiModelDescription
            fmiVersion="2.0"
            modelName="TestModel"
            guid="{12345678-1234-1234-1234-123456789012}">
        </fmiModelDescription>"""

        model_desc_path = temp_path / "modelDescription.xml"
        with open(model_desc_path, "w") as f:
            f.write(xml_content)

        # Test the function with directory path
        result = get_fmi_version(temp_path)
        assert result == "2.0"


def test_get_fmi_version_with_fmu_file():
    """Test get_fmi_version with an FMU file (ZIP archive)"""
    # Create a temporary FMU file (which is a ZIP archive)
    with tempfile.NamedTemporaryFile(suffix=".fmu", delete=False) as f:
        temp_fmu_path = f.name

    # Create a ZIP file with modelDescription.xml inside
    with zipfile.ZipFile(temp_fmu_path, "w") as zip_file:
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <fmiModelDescription
            fmiVersion="3.0"
            modelName="TestModel"
            instantiationToken="{12345678-1234-1234-1234-123456789012}">
        </fmiModelDescription>"""
        zip_file.writestr("modelDescription.xml", xml_content)

    try:
        # Test the function with FMU file - this should work as the function handles FMU files internally
        result = get_fmi_version(temp_fmu_path)
        assert result == "3.0"
    finally:
        # Clean up
        pathlib.Path(temp_fmu_path).unlink()


def test_get_fmi_version_missing_fmi_version_attribute():
    """Test get_fmi_version with XML missing fmiVersion attribute"""
    # Create XML without fmiVersion attribute
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function - should raise ValueError
        with pytest.raises(ValueError, match="fmiVersion attribute not found"):
            get_fmi_version(temp_xml_path)
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_get_fmi_version_unsupported_version():
    """Test get_fmi_version with unsupported FMI version"""
    # Create XML with unsupported FMI version
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="1.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function - should raise ValueError
        with pytest.raises(ValueError, match="Unsupported FMI version: 1.0"):
            get_fmi_version(temp_xml_path)
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_get_fmi_version_unsupported_file_type():
    """Test get_fmi_version with unsupported file type"""
    # Create a temporary file with unsupported extension
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"This is not an XML or FMU file")
        temp_txt_path = f.name

    try:
        # Test the function - should raise ValueError
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_fmi_version(temp_txt_path)
    finally:
        # Clean up
        pathlib.Path(temp_txt_path).unlink()


def test_read_model_description_fmi2():
    """Test read_model_description with FMI 2.0 content"""
    # Create a minimal FMI 2.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="2.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
        <ModelVariables>
            <ScalarVariable name="testVar" valueReference="0" causality="output">
                <Real start="1.0"/>
            </ScalarVariable>
        </ModelVariables>
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function
        result = read_model_description(temp_xml_path)
        assert result.fmi_version == "2.0"
        assert result.model_name == "TestModel"
        assert result.guid == "{12345678-1234-1234-1234-123456789012}"
        assert len(result.model_variables) == 1
        assert result.model_variables[0].name == "testVar"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_read_model_description_fmi3():
    """Test read_model_description with FMI 3.0 content"""
    # Create a minimal FMI 3.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="3.0"
        modelName="TestModel"
        instantiationToken="{12345678-1234-1234-1234-123456789012}">
        <ModelVariables>
            <Float64 name="testVar" valueReference="0" causality="output" start="1.0"/>
        </ModelVariables>
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function
        result = read_model_description(temp_xml_path)
        assert result.fmi_version == "3.0"
        assert result.model_name == "TestModel"
        assert result.instantiation_token == "{12345678-1234-1234-1234-123456789012}"
        # FMI 3.0 uses Variable wrapper, so we need to access the concrete variable
        assert len(result.model_variables) == 1
        var = result.model_variables[0]
        assert var.get_variable_type() == "Float64"
        assert var.concrete.name == "testVar"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_read_model_description_with_directory():
    """Test read_model_description with a directory containing modelDescription.xml"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create modelDescription.xml inside the directory
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <fmiModelDescription
            fmiVersion="2.0"
            modelName="TestModel"
            guid="{12345678-1234-1234-1234-123456789012}">
            <ModelVariables>
                <ScalarVariable name="testVar" valueReference="0" causality="output">
                    <Real start="1.0"/>
                </ScalarVariable>
            </ModelVariables>
        </fmiModelDescription>"""

        model_desc_path = temp_path / "modelDescription.xml"
        with open(model_desc_path, "w") as f:
            f.write(xml_content)

        # Test the function with directory path
        result = read_model_description(temp_path)
        assert result.fmi_version == "2.0"
        assert result.model_name == "TestModel"
        assert result.guid == "{12345678-1234-1234-1234-123456789012}"
        assert len(result.model_variables) == 1
        assert result.model_variables[0].name == "testVar"


def test_read_model_description_with_fmu():
    """Test read_model_description with an FMU file (ZIP archive)"""
    # Create a temporary FMU file (which is a ZIP archive)
    with tempfile.NamedTemporaryFile(suffix=".fmu", delete=False) as f:
        temp_fmu_path = f.name

    # Create a ZIP file with modelDescription.xml inside
    with zipfile.ZipFile(temp_fmu_path, "w") as zip_file:
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <fmiModelDescription
            fmiVersion="3.0"
            modelName="TestModel"
            instantiationToken="{12345678-1234-1234-1234-123456789012}">
            <ModelVariables>
                <Float64 name="testVar" valueReference="0" causality="output" start="1.0"/>
            </ModelVariables>
        </fmiModelDescription>"""
        zip_file.writestr("modelDescription.xml", xml_content)

    try:
        # Test the function with FMU file
        result = read_model_description(temp_fmu_path)
        assert result.fmi_version == "3.0"
        assert result.model_name == "TestModel"
        assert result.instantiation_token == "{12345678-1234-1234-1234-123456789012}"
        assert len(result.model_variables) == 1
        var = result.model_variables[0]
        assert var.get_variable_type() == "Float64"
        assert var.concrete.name == "testVar"
    finally:
        # Clean up
        pathlib.Path(temp_fmu_path).unlink()


def test_read_model_description_unsupported_version():
    """Test read_model_description with unsupported FMI version"""
    # Create XML with unsupported FMI version
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="1.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function - should raise ValueError
        with pytest.raises(ValueError, match="Unsupported FMI version: 1.0"):
            read_model_description(temp_xml_path)
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


def test_read_model_description_invalid_path():
    """Test read_model_description with invalid path"""
    # Test with a path that doesn't exist
    with pytest.raises(FileNotFoundError):
        read_model_description("/non/existent/path.xml")


def test_get_fmi_version_pathlib_path():
    """Test get_fmi_version with pathlib.Path object"""
    # Create a minimal FMI 2.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="2.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = pathlib.Path(f.name)

    try:
        # Test the function with pathlib.Path
        result = get_fmi_version(temp_xml_path)
        assert result == "2.0"
    finally:
        # Clean up
        temp_xml_path.unlink()


def test_read_model_description_pathlib_path():
    """Test read_model_description with pathlib.Path object"""
    # Create a minimal FMI 2.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="2.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
        <ModelVariables>
            <ScalarVariable name="testVar" valueReference="0" causality="output">
                <Real start="1.0"/>
            </ScalarVariable>
        </ModelVariables>
    </fmiModelDescription>"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = pathlib.Path(f.name)

    try:
        # Test the function with pathlib.Path
        result = read_model_description(temp_xml_path)
        assert result.fmi_version == "2.0"
        assert result.model_name == "TestModel"
        assert result.guid == "{12345678-1234-1234-1234-123456789012}"
        assert len(result.model_variables) == 1
        assert result.model_variables[0].name == "testVar"
    finally:
        # Clean up
        temp_xml_path.unlink()


def test_get_fmi_version_with_different_extensions():
    """Test get_fmi_version with different file extensions"""
    # Test with .fmu extension
    with tempfile.NamedTemporaryFile(suffix=".fmu", delete=False) as f:
        temp_fmu_path = f.name

    # Create a ZIP file with modelDescription.xml inside
    with zipfile.ZipFile(temp_fmu_path, "w") as zip_file:
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <fmiModelDescription
            fmiVersion="3.0"
            modelName="TestModel"
            instantiationToken="{12345678-1234-1234-1234-123456789012}">
        </fmiModelDescription>"""
        zip_file.writestr("modelDescription.xml", xml_content)

    try:
        # Test the function with FMU file
        result = get_fmi_version(temp_fmu_path)
        assert result == "3.0"
    finally:
        # Clean up
        pathlib.Path(temp_fmu_path).unlink()


def test_get_fmi_version_with_xml_extension():
    """Test get_fmi_version with .xml extension"""
    # Create a minimal FMI 2.0 XML content
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <fmiModelDescription
        fmiVersion="2.0"
        modelName="TestModel"
        guid="{12345678-1234-1234-1234-123456789012}">
    </fmiModelDescription>"""

    # Write to a temporary file with .xml extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        # Test the function
        result = get_fmi_version(temp_xml_path)
        assert result == "2.0"
    finally:
        # Clean up
        pathlib.Path(temp_xml_path).unlink()


@pytest.mark.parametrize(
    "fmu_path",
    [
        "2.0/Feedthrough.fmu",
        "2.0/BouncingBall.fmu",
        "2.0/VanDerPol.fmu",
        "2.0/Dahlquist.fmu",
        "2.0/Stair.fmu",
        "2.0/Resource.fmu",
        "3.0/Feedthrough.fmu",
        "3.0/BouncingBall.fmu",
        "3.0/VanDerPol.fmu",
        "3.0/Dahlquist.fmu",
        "3.0/Stair.fmu",
        "3.0/Resource.fmu",
    ],
)
def test_fmi_module_consistency_with_reference_fmus(fmu_path, reference_fmus_dir):
    """Test that get_fmi_version and read_model_description are consistent"""
    filename = (reference_fmus_dir / fmu_path).absolute()

    # Get version using get_fmi_version
    version_result = get_fmi_version(filename)

    # Get version from parsed model description
    model_result = read_model_description(filename)
    parsed_version = model_result.fmi_version

    # Both methods should return the same version
    assert version_result == parsed_version
