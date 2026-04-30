import importlib.machinery
import importlib.util
import io
import os
import sys
import tempfile
import textwrap
import pytest

XEED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed"))

loader = importlib.machinery.SourceFileLoader("xeed", XEED_PATH)
spec = importlib.util.spec_from_loader("xeed", loader)
_xeed = importlib.util.module_from_spec(spec)
loader.exec_module(_xeed)

BLOB_CLS = _xeed.BLOB_CLS
CONFIG_CLS = _xeed.CONFIG_CLS
Tool = _xeed.Tool
Formatter = _xeed.Formatter
ResolvingFormatter = _xeed.ResolvingFormatter
ReResolvingFormatter = _xeed.ReResolvingFormatter
HashedCache = _xeed.HashedCache
ToolChest = _xeed.ToolChest


def test_blob_paths():
    blob = BLOB_CLS({"zero": {"one": 1}})
    assert blob.get_path("zero.one") == 1
    assert list(blob.get_paths()) == ["zero", "zero.one"]

    blob = BLOB_CLS({"zero": {"one": 1, "two": 2}})
    assert blob.get_path("zero.one") == 1
    assert list(blob.get_paths()) == ["zero", "zero.one", "zero.two"]

    blob = BLOB_CLS({"zero": {"one": 1, "two": {"three": 3}}})
    assert list(blob.get_paths()) == ["zero", "zero.one", "zero.two", "zero.two.three"]
    assert blob.get_path("zero.one") == 1

def test_cfg_file_to_blob():
    contents = """
    [DEFAULT]
    [one]
    won: 1
    [one.two]
    three: 3
    """
    with tempfile.NamedTemporaryFile("w") as config_file:
        config_file.write(contents)
        config_file.flush()
        config = CONFIG_CLS.from_path(config_file.name)
    blob = config.to_blob()
    assert blob == {"one": {"won": "1", "two": {"three": "3"}}}
    assert list(blob.get_paths()) == ["one", "one.won", "one.two", "one.two.three"]
    assert blob.get_path("one.two.three") == "3"
    assert blob.get_path("one.two") == {"three": "3"}
    assert blob.get_path("one") == {"won": "1", "two": {"three": "3"}}

def test_cfg_dir_to_blob():
    contents = """
    [DEFAULT]
    [one]
    won: 1
    [one.two]
    three: 3
    """
    with tempfile.TemporaryDirectory() as config_dir:
        with open(f"{config_dir}/x.cfg", "w") as config_file:
            config_file.write(contents)
        config = CONFIG_CLS.from_path(config_file.name)
    blob = config.to_blob()
    assert blob == {"one": {"won": "1", "two": {"three": "3"}}}
    assert blob.get_path("one.two.three") == "3"
    assert blob.get_path("one.two") == {"three": "3"}
    assert blob.get_path("one") == {"won": "1", "two": {"three": "3"}}

def test_cfg_dir_x_to_blob():
    contents_x = """
    [DEFAULT]
    [one]
    won: 1
    """
    with tempfile.TemporaryDirectory() as config_dir:
        with open(f"{config_dir}/x.cfg", "w") as config_x:
            config_x.write(contents_x)
        config = CONFIG_CLS.from_path(config_dir)
    blob = config.to_blob()
    assert blob == {"one": {"won": "1"}}
    assert blob.get_path("one.won") == "1"
    assert blob.get_path("one") == {"won": "1"}

def test_cfg_dir_y_to_blob():
    contents_y = """
    [one.two]
    three: 3
    """
    with tempfile.TemporaryDirectory() as config_dir:
        with open(f"{config_dir}/y.cfg", "w") as config_y:
            config_y.write(contents_y)
        config = CONFIG_CLS.from_dir(config_dir)
    blob = config.to_blob()
    assert blob == {"one": {"two": {"three": "3"}}}
    assert blob.get_path("one.two.three") == "3"
    assert blob.get_path("one.two") == {"three": "3"}
    assert blob.get_path("one") == {"two": {"three": "3"}}

def test_cfg_dir_xy_to_blob():
    contents_x = """
    [DEFAULT]
    [one]
    won: 1
    """
    contents_y = """
    [one.two]
    three: 3
    """
    with tempfile.TemporaryDirectory() as config_dir:
        with open(f"{config_dir}/x.cfg", "w") as config_x:
            config_x.write(contents_x)
        with open(f"{config_dir}/y.cfg", "w") as config_y:
            config_y.write(contents_y)
        config = CONFIG_CLS.from_dir(config_dir)
    assert config.sections()
    blob = config.to_blob()
    assert blob == {"one": {"won": "1", "two": {"three": "3"}}}
    assert blob.get_path("one.two.three") == "3"
    assert blob.get_path("one.two") == {"three": "3"}
    assert blob.get_path("one") == {"won": "1", "two": {"three": "3"}}

def test_formatter():
    obj = Formatter()
    assert obj.format("This {--is-empty=x}", x=None) == "This "
    assert obj.format("This {--is-not=y}", y="not") == "This --is-not=not"

def test_resolving_formatter():
    X = BLOB_CLS.from_dict({"a": {"b": {"c": 0}}})
    resolver = lambda x, y: x.get_path(y)
    obj = ResolvingFormatter(resolver)
    assert obj.format("This is {a.b.c}", X) == "This is 0"

def test_reresolving_formatter():
    X = BLOB_CLS.from_dict({
        "a": {
            "bb": 1,
            "b": {
                "c": 0,
                "d": "{a.b.c}",
                "e": "{a.bb}"
            }}
    })
    resolver = lambda x, y: x.get_path(y)
    obj = ReResolvingFormatter(resolver)
    assert obj.format("This is {a.b.c}", X) == "This is 0"
    assert obj.format("This is {a.b.d}", X) == "This is 0"
    assert obj.format("This is {a.b.e}", X) == "This is 1"

def test_hashed_cache():
    blob = BLOB_CLS.empty()
    with tempfile.TemporaryDirectory() as tmpdir:
        blob.set_path("xeed.vars.hashfile", f"{tmpdir}/hash")
        blob.set_path("xeed.vars.cachedir", f"{tmpdir}/cache")
        cache = HashedCache.from_blob(blob)

        hash_1 = cache.blob_hash
        hash_2 = cache.blob_hash
        assert hash_1 == hash_2

        blob.set_path("a.b.c", 1)
        hash_3 = cache.blob_hash
        assert hash_3 != hash_2

@pytest.fixture
def mock_argv():
    original_argv = sys.argv
    mock = []
    sys.argv = mock
    yield mock
    sys.argv = original_argv

PARSER_CLASSES = [CONFIG_CLS]

@pytest.mark.parametrize("parser_cls", PARSER_CLASSES)
def test_multiline_code_in_cfg(parser_cls):
    mock_file_content = """
[xeed.types.tool.subtypes.testtool]
code:
    : class MyTestClass(Tool):
    :     def run(self):
    :         return "yay!"
"""
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(mock_file_content)
        tmp.flush()
        config = parser_cls()
        config.read(tmp.name)
        raw_code = config.get('xeed.types.tool.subtypes.testtool', 'code')
        assert raw_code == '\nclass MyTestClass(Tool):\n    def run(self):\n        return "yay!"'
        scope = {}
        exec(raw_code, {"Tool": Tool}, scope)
        obj = scope['MyTestClass'].from_blob({})
        assert obj.run() == "yay!"

@pytest.mark.parametrize("parser_cls", PARSER_CLASSES)
def test_toolchest(parser_cls):
    mock_file_content = """
[xeed.types.tool.subtypes.testtool]
code:
    : class MyTestClass(Tool):
    :     def run(self):
    :         return 0

[xeed.tools.foo.cmds.test]
type: xeed.types.tool.subtypes.testtool
"""
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(mock_file_content)
        tmp.flush()
        config = parser_cls()
        config.read(tmp.name)
        toolchest = ToolChest.from_blob(config.to_blob())
        assert len(list(toolchest._types)) == 1
        assert isinstance(toolchest.types, dict)
        assert len(toolchest.types) == 1
        assert 'xeed.types.tool.subtypes.testtool' in toolchest.types
        ttype = toolchest.types['xeed.types.tool.subtypes.testtool']
        assert issubclass(ttype, Tool)
        assert len(list(toolchest._tools)) == 1
        assert isinstance(toolchest.tools, dict)
        assert len(list(toolchest.tools)) == 1
        assert 'test' in toolchest.tools
        assert toolchest.tools["test"].run() == 0
