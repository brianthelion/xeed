import base64
import importlib.machinery
import importlib.util
import io
import json
import os
import shutil
import sys
import pytest
from unittest.mock import patch

XEED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed"))
XEED_D_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed.d"))
REAL_XEED_CFG = os.path.join(XEED_D_PATH, "xeed.cfg")
REAL_SELF_CFG = os.path.join(XEED_D_PATH, "self.cfg")

FAKE_ORIGIN_HASH = "a" * 40
FAKE_HEAD_HASH = "b" * 40

ORIGIN_XEED = b"#! /usr/bin/env python3\nXEED_ORIGIN_HASH = \"$Format:%H$\"\n"
HEAD_XEED = b"#! /usr/bin/env python3\nXEED_ORIGIN_HASH = \"$Format:%H$\"\n# updated\n"

def _real(filename):
    path = os.path.join(XEED_D_PATH, filename)
    with open(path, "rb") as f:
        return f.read()

ORIGIN_XEED_AT_HASH = ORIGIN_XEED.replace(b'"$Format:%H$"', f'"{FAKE_ORIGIN_HASH}"'.encode())

CANONICAL_FILES = {
    "xeed": (ORIGIN_XEED_AT_HASH, HEAD_XEED),
    "xeed.d/xeed.cfg": (lambda: _real("xeed.cfg"), lambda: _real("xeed.cfg")),
    "xeed.d/self.cfg": (lambda: _real("self.cfg"), lambda: _real("self.cfg")),
}


def fake_response(data):
    body = json.dumps(data).encode()
    resp = io.BytesIO(body)
    resp.status = 200
    resp.read = resp.read
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    return resp


def make_fake_urlopen(origin_hash=FAKE_ORIGIN_HASH, head_hash=FAKE_HEAD_HASH, dirty_files=None):
    dirty_files = dirty_files or set()

    def fake_urlopen(request):
        url = request.full_url if hasattr(request, 'full_url') else request

        # GET HEAD sha
        if "/git/refs/heads/main" in url:
            return fake_response({"object": {"sha": head_hash}})

        # GET file contents at a ref
        for path, (origin_content, head_content) in CANONICAL_FILES.items():
            if f"/contents/{path}?" in url:
                if f"ref={origin_hash}" in url:
                    content = origin_content() if callable(origin_content) else origin_content
                elif f"ref={head_hash}" in url or "ref=main" in url:
                    content = head_content() if callable(head_content) else head_content
                else:
                    raise AssertionError(f"Unexpected ref in URL: {url}")
                return fake_response({
                    "content": base64.b64encode(content).decode(),
                    "sha": "fileshaabc123",
                    "encoding": "base64",
                })

        raise AssertionError(f"Unexpected API call: {url}")

    return fake_urlopen


@pytest.fixture(scope="module")
def xeed():
    loader = importlib.machinery.SourceFileLoader("xeed", XEED_PATH)
    spec = importlib.util.spec_from_loader("xeed", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture
def project_dir(tmp_path, xeed, monkeypatch):
    xeed_d = tmp_path / "xeed.d"
    xeed_d.mkdir()
    xeed_script = tmp_path / "xeed"
    xeed_script.write_bytes(ORIGIN_XEED.replace(
        b'"$Format:%H$"', f'"{FAKE_ORIGIN_HASH}"'.encode()
    ))
    shutil.copy(REAL_XEED_CFG, xeed_d / "xeed.cfg")
    shutil.copy(REAL_SELF_CFG, xeed_d / "self.cfg")
    xeed.ENV["PWD"] = str(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    yield tmp_path
    xeed.ENV["PWD"] = os.environ.get("PWD", "")


def run_cmd(xeed, project_dir, cmd, *args):
    orig_argv = sys.argv
    orig_config = xeed.Cli.OPTIONS["--config"]
    try:
        sys.argv = [str(project_dir / "xeed"), "--config", str(project_dir / "xeed.d"), cmd, *args]
        xeed.Cli.OPTIONS["--config"] = None
        return xeed.main()
    except SystemExit as e:
        return e.code if e.code != 0 else 0
    finally:
        sys.argv = orig_argv
        xeed.Cli.OPTIONS["--config"] = orig_config


# --- self/pull tests ---

def test_pull_clean_updates_files(xeed, project_dir):
    with patch("urllib.request.urlopen", make_fake_urlopen()):
        result = run_cmd(xeed, project_dir, "self/pull")
    assert result == 0 or result is None
    assert (project_dir / "xeed").read_bytes() == HEAD_XEED.replace(b'"$Format:%H$"', f'"{FAKE_HEAD_HASH}"'.encode())

def test_pull_clean_updates_hash(xeed, project_dir):
    with patch("urllib.request.urlopen", make_fake_urlopen()):
        run_cmd(xeed, project_dir, "self/pull")
    xeed_content = (project_dir / "xeed").read_text()
    assert FAKE_HEAD_HASH in xeed_content

def test_pull_dirty_aborts(xeed, project_dir):
    original = (project_dir / "xeed.d" / "self.cfg").read_bytes()
    (project_dir / "xeed.d" / "self.cfg").write_bytes(original + b"# local modification\n")
    with patch("urllib.request.urlopen", make_fake_urlopen()):
        result = run_cmd(xeed, project_dir, "self/pull")
    assert result != 0

def test_pull_dirty_does_not_overwrite(xeed, project_dir):
    original = (project_dir / "xeed.d" / "self.cfg").read_bytes()
    modified = original + b"# local modification\n"
    (project_dir / "xeed.d" / "self.cfg").write_bytes(modified)
    with patch("urllib.request.urlopen", make_fake_urlopen()):
        run_cmd(xeed, project_dir, "self/pull")
    assert (project_dir / "xeed.d" / "self.cfg").read_bytes() == modified
