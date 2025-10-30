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

assert sys.version_info >= (3, 10, 12)

LOG = logging.getLogger(__name__)
PATH = os.path.abspath(__file__)
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

class StrJoin(argparse.Action):
    def __call__(self, parser, namespace, values, *_):
        setattr(namespace, self.dest, " ".join(values))

class Cli:
    OPTIONS = {
        "--config": "xeed.d",
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

    def extend(self, subcmd, cmdstr):
        LOG.debug(f"{subcmd} {cmdstr}")
        parser = self._subparsers.add_parser(subcmd, add_help=False)
        parser.set_defaults(cmdstr=cmdstr)
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

    def get_path(self, dotted):
        token, *remainder = self._split(dotted)
        LOG.debug(remainder)
        item = self[token]
        if len(remainder) == 0:
            return item
        sub = self.__class__.from_dict(item)
        return sub.get_path(self._join(remainder))

    def set_path(self, dotted, value):
        token, *remainder = self._split(dotted)
        if len(remainder) == 0:
            return self.__setitem__(token, value)
        if token not in self:
            self[token] = self.__class__.from_dict({})
        path = self._join(remainder)
        return self[token].set_path(path, value)

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

    @property
    def cache_dir(self):
        return self.config_blob.get_path("DEFAULT.cachedir")

    @property
    def files(self):
        for section_name in self.config_blob.get("file", {}).keys():
            yield self.config_blob.get_path(f"file.{section_name}")

    def write(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # config_blob = self.config_blob
        # for section_name in config_blob.get("file", {}).keys():
        #     self._write_one(section_name, config_blob)
        for file_blob in self.files:
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

    @property
    def hash_path(self):
        return self.config_blob.get_path("DEFAULT.hashfile")

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

class ToolChest:

    def __init__(self, config_blob):
        self._config_blob = config_blob

    @property
    def tools(self):
        return self._config_blob.get("tool", {})

    @classmethod
    def from_blob(cls, config_blob):
        return cls(config_blob)


FORMATTER = ReResolvingFormatter(lambda x, y: x.get_path(y))
BLOB_CLS = Blob
CONFIG_CLS = CfgConfig
CONFIG_CLS.BLOB_CLS = BLOB_CLS
CACHE_CLS = XeedCache
CACHE_CLS.FORMATTER = FORMATTER
CLI_CLS = Cli
CLI_CLS.FORMATTER = Formatter()

def main():

    blob = BLOB_CLS.from_dict(dict(env=ENV, user=USER))

    cli = CLI_CLS.empty()
    cli.parse(final=False)
    logging.basicConfig(level=cli.log_level)

    try:
        config = CONFIG_CLS.from_path(cli.config_path)
    except FileNotFoundError as err:
        return str(err)
    blob.update(config.to_blob())

    toolchest = ToolChest.from_blob(blob)
    if not toolchest.tools:
        return f"Config {cli.config_path} must have at least one [tool.mytool] section!"

    for tool_name, tool_blob in toolchest.tools.items():
        tool_cmd = tool_blob.get("cmd", None) or tool_name
        tool_call = tool_blob.get("cmdstr")
        cli.extend(tool_cmd, cmdstr=tool_call)

    cli.extend("help", cmdstr=None)
    cli.parse(final=True)
    if cli.check("help"):
    #     cli.print_help(cache.blob_hash)
        return 0

    cache = CACHE_CLS.from_blob(blob)
    blob.update(xeed=dict(PATH=PATH, HASH=cache.blob_hash, PREFIX=cli.prefix))
    blob.update(cli=cli.to_dict())

    cache.write()
    cmdstr = FORMATTER.format(blob.get_path(f"cli.cmdstr"), blob)
    LOG.info(cmdstr)
    return subprocess.call(cmdstr, shell=True)

if __name__ == "__main__":
    exit(main())

import pytest

def test_blob_paths():
    blob = BLOB_CLS({"zero": {"one": 1}})
    assert blob.get_path("zero.one") == 1

    blob = BLOB_CLS({"zero": {"one": 1, "two": 2}})
    assert blob.get_path("zero.one") == 1

    blob = BLOB_CLS({"zero": {"one": 1, "two": {"three": 3}}})
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
        blob.set_path("DEFAULT.hashfile", f"{tmpdir}/hash")
        blob.set_path("DEFAULT.cachedir", f"{tmpdir}/cache")
        cache = HashedCache.from_blob(blob)

        hash_1 = cache.blob_hash
        hash_2 = cache.blob_hash
        assert hash_1 == hash_2

        blob.set_path("a.b.c", 1)
        hash_3 = cache.blob_hash
        assert hash_3 != hash_2

        # old_hash = cache.old_hash
        # new_hash = cache.new_hash
        # # tmp_hash = cache.new_hash
        # assert old_hash == CACHE_CLS.EMPTY_HASH
        # assert old_hash == cache.old_hash
        # # assert new_hash != old_hash
        # # assert tmp_hash == new_hash
        # assert new_hash == cache.new_hash
        # assert new_hash != old_hash
        # cache.update()
        # old_hash = cache.old_hash
        # new_hash = cache.new_hash
        # assert old_hash == new_hash
        # assert new_hash == tmp_hash
        # tmp_hash = new_hash
        # blob.set_path("a.b.c", 1)
        # old_hash = cache.old_hash
        # new_hash = cache.new_hash
        # assert old_hash != new_hash
        # assert old_hash == tmp_hash
        # cache.update()

@pytest.fixture
def mock_argv():
    original_argv = sys.argv
    mock = []
    sys.argv = mock
    yield mock
    sys.argv = original_argv

def test_cli_args_help(mock_argv):
    mock_argv.extend(["./xeed.py", "help"])
    assert main() == 0

