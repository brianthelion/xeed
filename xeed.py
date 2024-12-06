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

class StrJoin(argparse.Action):
    def __call__(self, parser, namespace, values, *_):
        setattr(namespace, self.dest, " ".join(values))

class Cli:
    OPTIONS = {
        "--config": "xeed.cfg",
        "--log-level": "ERROR",
    }

    @classmethod
    def empty(cls):
        parser = argparse.ArgumentParser()
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
        obj = Formatter()
        return obj.format("{path} {--config=x} {--log-level=y}",
                          path=PATH, x=ns.config, y=ns.log_level)

    def extend(self, subcmd, cmdstr):
        LOG.debug(f"{subcmd} {cmdstr}")
        parser = self._subparsers.add_parser(subcmd)
        parser.set_defaults(cmdstr=cmdstr)
        parser.add_argument("extra_args",
                            nargs=argparse.REMAINDER,
                            action=StrJoin,
                            )

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
    BLOB_CLS = Blob

    @classmethod
    def from_path(cls, path_str):
        LOG.debug(f"Loading config {path_str}")
        with open(path_str, "r") as afile:
            return cls.from_file(afile)

    @classmethod
    def from_file(cls, filelike):
        self = cls()
        self.read_file(filelike)
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

class CacheManager:
    READ_SIZE = 8192
    HASH_SIZE = 8
    HASH_CLS = hashlib.sha256
    CONFIG_CLS = CfgConfig
    EMPTY_HASH = ""

    def __init__(self, config_blob):
        self._config_blob = config_blob

    @classmethod
    def compute_hash(cls, path):
        hash_obj = cls.HASH_CLS()
        with open(path, 'rb') as f:
            while chunk := f.read(cls.READ_SIZE):
                hash_obj.update(chunk)
        hash_str = hash_obj.hexdigest()
        return hash_str[:cls.HASH_SIZE]

    @property
    def config_blob(self):
        return self._config_blob

    @property
    def cache_dir(self):
        return self.config_blob \
                   .get_path("DEFAULT.cachedir")

    @property
    def hash_path(self):
        hash_file = self.config_blob.get_path("DEFAULT.hashfile")
        return os.path.join(self.cache_dir, hash_file)

    @property
    def new_hash(self):
        with tempfile.NamedTemporaryFile("w") as tmp:
            blob_str = json.dumps(self.config_blob, sort_keys=True)
            tmp.write(blob_str)
            tmp.flush()
            return self.compute_hash(tmp.name)

    @property
    def old_hash(self):
        if not os.path.exists(self.hash_path):
            return self.EMPTY_HASH
        return self.read_hash()

    def write_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        config_blob = self.config_blob
        for section_name in config_blob.get("file", {}).keys():
            self._write_one(section_name, config_blob)

    def _write_one(self, blob_name, config_blob):
        file_path = config_blob.get_path(f"file.{blob_name}.path")
        file_path = FORMATTER.format(file_path, config_blob)
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_contents = config_blob.get_path(f"file.{blob_name}.contents")
        file_contents = FORMATTER.format(file_contents, config_blob)
        with open(file_path, "w") as open_file:
            open_file.write(file_contents)

    def read_hash(self):
        with open(self.hash_path, "r") as hash_file:
            return hash_file.read().strip()

    def write_hash(self, new_hash):
        with open(self.hash_path, "w") as hash_file:
            hash_file.write(new_hash)

    def update(self):
        if self.new_hash == self.old_hash:
            return False
        self.write_cache()
        self.write_hash(self.new_hash)
        return True

    @classmethod
    def from_path(cls, config_path):
        raise NotImplementedError()

    @classmethod
    def from_blob(cls, config_blob):
        return cls(config_blob)


FORMATTER = ReResolvingFormatter(lambda x, y: x.get_path(y))
CONFIG_CLS = CfgConfig

def main():

    cli = Cli.empty()
    cli.parse(final=False)
    logging.basicConfig(level=cli.log_level)

    try:
        blob = CONFIG_CLS \
            .from_path(cli.config_path) \
            .to_blob()
    except FileNotFoundError as err:
        exit(str(err))

    cache = CacheManager.from_blob(blob)

    if blob.get("tool", None) is None:
        exit(f"Config {cli.config_path} must have at least one [tool.mytool] section!")

    for tool_name in blob.get("tool", {}).keys():
        path = f"tool.{tool_name}"
        cli.extend(tool_name,
                   cmdstr=blob.get_path(f"{path}.cmdstr"))

    cli.parse(final=True)
    blob.update(cli=cli.to_dict(),
                env=ENV,
                xeed=dict(PATH=PATH, HASH=cache.new_hash, PREFIX=cli.prefix),
                user=USER)
    cache.update()

    cmdstr = FORMATTER.format(blob.get_path(f"cli.cmdstr"), blob)
    LOG.info(cmdstr)
    return subprocess.call(cmdstr, shell=True)

if __name__ == "__main__":
    exit(main())

import pytest
import io

def test_blob_paths():
    blob = Blob({"zero": {"one": 1}})
    assert blob.get_path("zero.one") == 1

    blob = Blob({"zero": {"one": 1, "two": 2}})
    assert blob.get_path("zero.one") == 1

    blob = Blob({"zero": {"one": 1, "two": {"three": 3}}})
    assert blob.get_path("zero.one") == 1

def test_cfg_to_blob():
    filelike = io.StringIO("""
    [DEFAULT]
    [one]
    [one.two]
    three: 3
    """)
    config = CfgConfig.from_file(filelike)
    blob = config.to_blob()
    assert blob.get_path("one.two.three") == "3"
    assert blob.get_path("one.two") == {"three": "3"}
    assert blob.get_path("one") == {"two": {"three": "3"}}

def test_formatter():
    obj = Formatter()
    assert obj.format("This {--is-empty=x}", x=None) == "This "
    assert obj.format("This {--is-not=y}", y="not") == "This --is-not=not"

def test_resolving_formatter():
    X = Blob.from_dict({"a": {"b": {"c": 0}}})
    resolver = lambda x, y: x.get_path(y)
    obj = ResolvingFormatter(resolver)
    assert obj.format("This is {a.b.c}", X) == "This is 0"

def test_reresolving_formatter():
    X = Blob.from_dict({
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
