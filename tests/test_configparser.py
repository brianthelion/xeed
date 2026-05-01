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
    Documents that ConfigParser strips all leading whitespace from multiline
    values, flattening indentation. This is why xeed uses the ': ' prefix
    convention to preserve Python indentation in code blocks.
    """
    mock_cfg = (
        "[debug]\n"
        "code =\n"
        "    class DebugClass:\n"
        "        def run(self):\n"
        "            pass"
    )

    config = ParserClass()
    config.read_string(mock_cfg)

    raw_val = config.get('debug', 'code')
    lines = raw_val.splitlines()

    if not lines[0].strip():
        lines.pop(0)

    class_indent = len(lines[0]) - len(lines[0].lstrip())
    method_indent = len(lines[1]) - len(lines[1].lstrip())

    assert class_indent == 0, "ConfigParser should strip all leading whitespace"
    assert method_indent == 0, "ConfigParser should flatten indentation"


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
