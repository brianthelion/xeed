#! /usr/bin/env python3

import sys
import argparse
import configparser
import functools
import subprocess
import os
import types
import logging
import string
import socket
import hashlib
import json
import tempfile
import getpass
import glob
#import fnmatch
import re
import textwrap

assert sys.version_info >= (3, 10, 12)

LOG = logging.getLogger(__name__)
PATH = os.path.abspath(__file__)
HERE = os.path.dirname(PATH)

DEFAULT_CONFIG = "xeed.cfg"

ENV = dict(os.environ)
ENV["HOSTNAME"] = socket.gethostname()
ENV["FQDN"] = socket.getfqdn()

USER = dict()
USER["UID"] = os.getuid()
USER["GID"] = os.getgid()
USER["NAME"] = getpass.getuser()


def exit(*args, **kwargs):
    return sys.exit(*args, **kwargs)

def log(level=logging.DEBUG):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log the function call with its arguments
            LOG.log(level, "Calling %s(args=%r, kwargs=%r)", func.__name__, args, kwargs)
            try:
                # Execute the original function
                result = func(*args, **kwargs)
                # Log the return value
                LOG.log(level, "%s returned %r", func.__name__, result)
                return result
            except Exception as e:
                # Exceptions are always logged at ERROR level with a traceback
                LOG.exception("Exception in %s: %s", func.__name__, e)
                # Re-raise the exception
                raise
        return wrapper
    return decorator

def find_file_dir(filename):
    pattern = f"{HERE}/**/{filename}"
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    nearest = min(matches, key=len)
    return os.path.dirname(nearest)

def as_dict(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return dict(func(*args, **kwargs))
    return wrapper

class StrJoin(argparse.Action):
    def __call__(self, parser, namespace, values, *_):
        setattr(namespace, self.dest, " ".join(values))

class Cli:
    OPTIONS = {
        "--config": find_file_dir(DEFAULT_CONFIG),
        "--log-level": "ERROR",
        "--use-hash": None,
    }
    PREFIX_TEMPLATE = "{path} {--config=x} {--log-level=y} {--use-hash=z}"
    FORMATTER = None

    @classmethod
    def empty(cls):
        parser = argparse.ArgumentParser(add_help=False)
        for k, v in cls.OPTIONS.items():
            parser.add_argument(k, default=v)
        return cls(parser)

    def __init__(self, parser):
        self._parser = parser
        self._ns = None
        self._remainder = None

    def parse(self, final=True):
        _parser = self._parser
        kwargs = {"args": self._remainder,
                  "namespace": self._ns,
                  }
        if final:
            self._ns = _parser.parse_args(**kwargs)
        else:
            self._ns, self._remainder = \
                _parser.parse_known_args(**kwargs)

    def to_dict(self):
        return vars(self._ns)

    def print_help(self, new_hash, file=None):
        self._parser.print_help(file=file)
        print(file=file)
        print(f"Current xeed hash: {new_hash}", file=file)

    @functools.cached_property
    def vars(self):
        return self._ns

    @functools.cached_property
    def config_path(self):
        return self._ns.config

    @functools.cached_property
    def log_level(self):
        return self._ns.log_level

    @functools.cached_property
    def _subparsers(self):
        return self._parser.add_subparsers(dest="sub", required=True)

    @functools.cached_property
    def prefix(self):
        ns = self._ns
        obj = self.FORMATTER
        return obj.format(self.PREFIX_TEMPLATE,
                          path=PATH, x=ns.config, y=ns.log_level,
                          z=ns.use_hash)

    def extend(self, subcmd, **kwargs):
        LOG.debug(f"{subcmd} {kwargs}")
        parser = self._subparsers.add_parser(subcmd, add_help=False)
        parser.set_defaults(foo="bar", **kwargs)
        parser.add_argument("extra_args",
                            nargs=argparse.REMAINDER,
                            action=StrJoin,
                            )

    def check(self, sub):
        return self._ns is not None \
            and self._ns.sub == sub

class Blob(dict):
    DELIM = "."

    @classmethod
    def _split(cls, dotted):
        return dotted.split(cls.DELIM)

    @classmethod
    def _join(cls, tokens):
        return cls.DELIM.join(tokens)

    def get_path(self, dotted_path):
        token, *remainder = self._split(dotted_path)
        LOG.debug(remainder)
        try:
            item = self[token]
        except KeyError:
            raise KeyError(dotted_path)
        if len(remainder) == 0:
            return item
        sub = self.__class__.from_dict(item)
        try:
            return sub.get_path(self._join(remainder))
        except KeyError as err:
            raise KeyError(dotted_path)

    def set_path(self, dotted_path, value):
        token, *remainder = self._split(dotted_path)
        if len(remainder) == 0:
            return self.__setitem__(token, value)
        if token not in self:
            self[token] = self.__class__.from_dict({})
        path = self._join(remainder)
        return self[token].set_path(path, value)

    def get_paths(self, prefix=None):
        for key, value in self.items():
            next_prefix = key if prefix is None else f"{prefix}.{key}"
            yield next_prefix
            if isinstance(value, dict):
                blob = self.__class__.from_dict(value)
                yield from blob.get_paths(prefix=next_prefix)

    def set_paths(self, path_map):
        for dotted_path, value in path_map.items():
            self.set_path(dotted_path, value)

    def merge(self, other_blob):
        for dotted_path in other_blob.get_paths():
            value = other_blob.get_path(dotted_path)
            if isinstance(value, dict):
                continue
            self.set_path(dotted_path, value)

    @classmethod
    def from_dict(cls, adict):
        LOG.debug(adict)
        assert isinstance(adict, dict)
        return cls(adict)

    @classmethod
    def empty(cls):
        return cls.from_dict({})

class CfgConfig(configparser.ConfigParser):
    BLOB_CLS = None
    MULTILINE_SEP = "\n:"

    @classmethod
    def from_path(cls, path_str):
        LOG.debug(f"Loading config {path_str}")
        if not os.path.exists(path_str):
            exit(f"Config path {path_str} does not exist!")
        elif os.path.isfile(path_str):
            return cls.from_file(path_str)
        elif os.path.isdir(path_str):
            return cls.from_dir(path_str)
        else:
            raise NotImplementedError()

    @classmethod
    def from_dir(cls, path_str):
        assert os.path.exists(path_str), path_str
        assert os.path.isdir(path_str), path_str
        config_paths = (f"{path_str}/{p}" for p in os.listdir(path_str))
        self = cls()
        self.read(config_paths)
        return self

    @classmethod
    def from_file(cls, path_str):
        assert os.path.exists(path_str), path_str
        assert os.path.isfile(path_str), path_str
        self = cls()
        self.read(path_str)
        return self

    def to_blob(self):
        out = self.BLOB_CLS.empty()
        for section_name, section_dict in self.items():
            for key, value in section_dict.items():
                out.set_path(f"{section_name}.{key}", value)
        return out

    def _read(self, fp, fpname):
        retval = super()._read(fp, fpname)
        self._postproc(self._sections)
        self._postproc(self._defaults)
        return retval

    @classmethod
    def _postproc(cls, section_dict):
        for k, v in section_dict.items():
            for kk, vv in v.items():
                LOG.debug(f"BEFORE {v[kk]}")
                v[kk] = cls._clean_multiline(vv)
                LOG.debug(f"AFTER {v[kk]}")

    @classmethod
    def _clean_multiline(cls, value):
        assert isinstance(value, str), f"ERROR: Found {value} of type {type(value)}"
        LOG.debug(f"VALUE {value}")
        sep = cls.MULTILINE_SEP
        if sep not in value:
            return value
        return textwrap.dedent(value.replace(sep, "\n"))

class TomlConfig:
    pass

class Formatter(string.Formatter):
    EMPTY_VALS = [None, ""]

    def get_value(self, key, args, kwargs):
        if "=" not in key:
            return super().get_value(key, args, kwargs)
        prefix, key = key.split("=")
        value = super().get_value(key, args, kwargs)
        if value in self.EMPTY_VALS:
            return ""
        return f"{prefix}={value}"

class ResolvingFormatter(string.Formatter):
    def __init__(self, resolver, *args, **kwargs):
        self._resolver = resolver

    def format(self, format_string, namespace):
        return super().format(format_string, namespace)

    def get_field(self, field_name, args, kwargs):
        LOG.debug(field_name)
        assert len(args) == 1
        assert len(kwargs) == 0
        namespace, *_ = args
        val = self._resolve(namespace, field_name)
        return val, field_name

    def _resolve(self, namespace, field_name):
        return self._resolver(namespace, field_name)

class ReResolvingFormatter(ResolvingFormatter):
    def _resolve(self, namespace, field_name):
        val = super()._resolve(namespace, field_name)
        if not isinstance(val, str):
            return val
        return self.format(val, namespace)

class CacheBase:
    CONFIG_CLS = None
    def __init__(self, config_blob):
        self._config_blob = config_blob

    @property
    def config_blob(self):
        return self._config_blob

    def write(self):
        raise NotImplementedError()

    @classmethod
    def from_path(cls, config_path):
        raise NotImplementedError()

    @classmethod
    def from_blob(cls, config_blob):
        return cls(config_blob)


class FileCache(CacheBase):
    CACHE_KEY = "xeed.vars.cachedir"
    SEP = "."
    FILE_REGEX = re.compile(r"xeed\.tools\.[a-zA-Z]+\.files\.[a-zA-Z/]+$")

    @property
    def cache_dir(self):
        return self.config_blob.get_path(self.CACHE_KEY)

    @property
    @as_dict
    def files(self):
        regex = self.FILE_REGEX
        blob = self._config_blob
        for key in blob.get_paths():
            if regex.match(key):
                yield key.split(self.SEP)[-1], blob.get_path(key)


    def write(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        for _, file_blob in self.files.items():
            self._write_one(file_blob, self.config_blob)

    @classmethod
    def _write_one(cls, file_blob, config_blob):
        file_path = file_blob.get_path("path")
        file_path = cls.FORMATTER.format(file_path, config_blob)
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_contents = file_blob.get_path("contents")
        file_contents = cls.FORMATTER.format(file_contents, config_blob)
        with open(file_path, "w") as open_file:
            open_file.write(file_contents)

class HashedCache(CacheBase):
    READ_SIZE = 8192
    HASH_SIZE = 8
    HASH_CLS = hashlib.sha256
    EMPTY_HASH = ""

    @property
    @log(level=logging.INFO)
    def blob_hash(self):
        with tempfile.NamedTemporaryFile("w") as tmp:
            blob_str = json.dumps(self.config_blob, sort_keys=True)
            tmp.write(blob_str)
            tmp.flush()
            return self.compute_hash(tmp.name)

    @classmethod
    def compute_hash(cls, path):
        hash_obj = cls.HASH_CLS()
        with open(path, 'rb') as f:
            while chunk := f.read(cls.READ_SIZE):
                hash_obj.update(chunk)
        hash_str = hash_obj.hexdigest()
        return hash_str[:cls.HASH_SIZE]

class SmartCache(HashedCache):
    HASH_KEY = "xeed.hashfile"

    @property
    def hash_path(self):
        return self.config_blob.get_path(self.HASH_KEY)

    @property
    def hash_fullpath(self):
        return os.path.join(self.cache_dir, hash_file)

    @property
    @log(level=logging.INFO)
    def disk_hash(self):
        if not os.path.exists(self.hash_fullpath):
            return self.EMPTY_HASH
        return self.read_hash()

    def read_hash(self):
        with open(self.hash_fullpath, "r") as hash_file:
            return hash_file.read().strip()

    def write_hash(self, blob_hash):
        with open(self.hash_fullpath, "w") as hash_file:
            hash_file.write(blob_hash)

class XeedCache(HashedCache, FileCache):
    pass

class Tool:
    def __init__(self, blob):
        self._config_blob = blob

    @property
    def cli(self):
        return self._config_blob.get("cmd", None)

    @classmethod
    def from_blob(cls, blob):
        return cls(blob)

class ToolChest:
    SEP = "."
    CMD_REGEX = re.compile(r"xeed\.tools\.[a-zA-Z]+\.cmds\.[a-zA-Z/]+$")
    TYPES_REGEX = re.compile(r"xeed\.types\.tool\.subtypes\.[a-zA-Z/]+$")

    def __init__(self, config_blob):
        self._config_blob = config_blob

    @property
    @as_dict
    def tools(self):
        blob = self._config_blob
        regex = self.CMD_REGEX
        for key, blob in self._walk(blob, regex):
            yield key.split(self.SEP)[-1], self._tool_factory(blob)

    @property
    @as_dict
    def types(self):
        for key, blob in self._types:
            yield key, self._type_factory(blob)

    @property
    def _tools(self):
        blob = self._config_blob
        regex = self.CMD_REGEX
        yield from self._walk(blob, regex)

    @property
    def _types(self):
        blob = self._config_blob
        regex = self.TYPES_REGEX
        yield from self._walk(blob, regex)

    @staticmethod
    def _walk(blob, regex):
        for key in blob.get_paths():
            if regex.match(key):
                yield key, blob.get_path(key)

    @classmethod
    def from_blob(cls, config_blob):
        return cls(config_blob)

    def _type_factory(self, blob):
        assert "code" in blob, blob
        code = blob["code"]
        out = {}
        exec(code, {"Tool": Tool, "LOG": LOG}, out)
        out = [v for v in out.values() if isinstance(v, type) and issubclass(v, Tool)]
        assert len(out) == 1, out
        return out.pop()

    def _tool_factory(self, blob):
        assert "type" in blob, blob
        type_name = blob["type"]
        return self.types[type_name].from_blob(blob)


FORMATTER = ReResolvingFormatter(lambda x, y: x.get_path(y))
BLOB_CLS = Blob
CONFIG_CLS = CfgConfig
CONFIG_CLS.BLOB_CLS = BLOB_CLS
CACHE_CLS = XeedCache
CACHE_CLS.FORMATTER = FORMATTER
CLI_CLS = Cli
CLI_CLS.FORMATTER = Formatter()

def main():

    blob = BLOB_CLS.empty() # just a nested dictionary with some helper methods
    blob.set_paths({"xeed.vars.env": ENV,
                    "xeed.vars.user": USER,
                    "xeed.vars.XEED_EXE": PATH})

    cli = CLI_CLS.empty()
    cli.parse(final=False)
    if cli.config_path is None:
        return f"Unable to locate config file '{DEFAULT_CONFIG}' under {PATH}!"

    logging.basicConfig(level=cli.log_level)

    try:
        config = CONFIG_CLS.from_path(cli.config_path)
    except FileNotFoundError as err:
        return str(err)
    blob.merge(config.to_blob())

    toolchest = ToolChest.from_blob(blob)
    if not toolchest.tools:
        return f"Config {cli.config_path} must have at least one [tool.mytool] section!"

    for tool_name, tool in toolchest.tools.items():
        cmd = tool.cli or tool_name
        cli.extend(cmd, tool=tool)

    cli.parse(final=True)
    cache = CACHE_CLS.from_blob(blob)
    blob.set_paths({"xeed.vars.HASH": cache.blob_hash,
                    "xeed.vars.PREFIX": cli.prefix})
    blob.set_path("xeed.vars.cli", cli.to_dict())
    cache.write()

    resolver = lambda x: FORMATTER.format(x, blob)
    return cli.vars.tool.run(**locals())

if __name__ == "__main__":
    exit(main())

import pytest
import io
import textwrap

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
            }}})
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

def test_cli_args_help(mock_argv, monkeypatch):
    mock_argv.extend(["./xeed.py", "self/help"])
    monkeypatch.setattr(sys, 'stdin', open(os.devnull))
    assert main() == 0

PARSER_CLASSES = [CfgConfig]

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
        # clean_code = textwrap.dedent(raw_code)
        scope = {}
        # exec(clean_code, {}, scope)
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
