import configparser
import textwrap
import pytest
import tempfile
import os

PARSER_CLASSES = [
    configparser.ConfigParser,
    configparser.RawConfigParser
]

@pytest.mark.parametrize("ParserClass", PARSER_CLASSES)
def test_verify_raw_configparser_behavior(ParserClass):
    """
    DEBUG TEST: This test isolates exactly what ConfigParser returns.
    If ConfigParser strips all whitespace, this assertion will show it.
    """
    # 1. Setup a file where the method is clearly indented deeper than the class
    # class -> 4 spaces
    # def   -> 8 spaces
    mock_cfg = (
        "[debug]\n"
        "code =\n"
        "    class DebugClass:\n"
        "        def run(self):\n"
        "            pass"
    )

    config = ParserClass()
    config.read_string(mock_cfg)

    # 2. Get the raw value
    raw_val = config.get('debug', 'code')

    print(f"\n[{ParserClass.__name__}] RAW VALUE REPR:\n{repr(raw_val)}")

    # 3. VERIFICATION
    # Split into lines to check indentation of specific lines
    lines = raw_val.splitlines()

    # Remove the first empty line if it exists (common in multiline ini values)
    if not lines[0].strip():
        lines.pop(0)

    class_line = lines[0]
    method_line = lines[1]

    # Calculate indentation
    class_indent = len(class_line) - len(class_line.lstrip())
    method_indent = len(method_line) - len(method_line.lstrip())

    print(f"Class Indent: {class_indent}")
    print(f"Method Indent: {method_indent}")

    # ConfigParser behavior rule:
    # It usually preserves the indentation found in the file.
    # If method_indent > class_indent, then ConfigParser respects relative whitespace.
    # If method_indent == class_indent, ConfigParser is flattening it.
    assert method_indent > class_indent, (
        f"ConfigParser flattened the indentation! Class: {class_indent}, Method: {method_indent}"
    )

@pytest.mark.parametrize("ParserClass", PARSER_CLASSES)
def test_staggered_indentation_executes_cleanly(ParserClass):
    """
    Verifies that if the .cfg file has 'staircase' indentation (method deeper than class),
    textwrap.dedent preserves the relative indentation required for Python execution.
    """
    # FIX: We construct the string explicitly to guarantee exact whitespace preservation,
    # avoiding issues where copy-pasting this file mangles triple-quote indentation.
    mock_cfg = (
        "[tool.success]\n"
        "code =\n"
        "    class ValidTool:\n"
        "        def run(self):\n"
        "            return 'it works'"
    )

    # 2. Parse it
    config = ParserClass()
    config.read_string(mock_cfg)

    # 3. Dedent and Execute
    raw_code = config.get('tool.success', 'code')
    clean_code = textwrap.dedent(raw_code)

    # Debug info
    # print(f"\nRAW CODE:\n{repr(raw_code)}")
    # print(f"\nCLEAN CODE:\n{clean_code}")

    scope = {}
    exec(clean_code, {}, scope)

    assert "ValidTool" in scope
    instance = scope["ValidTool"]()
    assert instance.run() == "it works"


@pytest.mark.parametrize("ParserClass", PARSER_CLASSES)
def test_colon_delimiter_with_tempfile(ParserClass):
    """
    Replicates the failing case but uses explicit string construction
    to ensure the file on disk has the correct indentation.
    """
    # FIX: Explicit construction again.
    # Class indented 4 spaces. Method indented 8 spaces.
    mock_file_content = (
        "[tool.test]\n"
        "code:\n"
        "    class MyTestClass:\n"
        "        def run(self):\n"
        "            return 'Indentation preserved!'"
    )

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        try:
            tmp.write(mock_file_content)
            tmp.close()

            config = ParserClass()
            config.read(tmp.name)

            raw_code = config.get('tool.test', 'code')
            clean_code = textwrap.dedent(raw_code)

            scope = {}
            exec(clean_code, {}, scope)

            assert scope['MyTestClass']().run() == "Indentation preserved!"
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)


@pytest.mark.parametrize("ParserClass", PARSER_CLASSES)
def test_flat_indentation_raises_indentation_error(ParserClass):
    """
    Verifies that 'flat' indentation (method aligned with class) causes the
    IndentationError.
    """
    # BAD INDENTATION (Intentional):
    # Both class and def are indented by 4 spaces.
    mock_cfg = (
        "[tool.fail]\n"
        "code =\n"
        "    class BrokenTool:\n"
        "    def run(self):\n"
        "        return 'crash'"
    )

    config = ParserClass()
    config.read_string(mock_cfg)

    clean_code = textwrap.dedent(config.get('tool.fail', 'code'))

    # This confirms your error is expected behavior for flat indentation
    with pytest.raises(IndentationError) as excinfo:
        exec(clean_code, {})

    assert "expected an indented block" in str(excinfo.value)
