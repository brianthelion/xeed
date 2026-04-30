import base64
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import urllib.error
import pytest
from unittest.mock import patch

XEED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed"))
XEED_D_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed.d"))

FAKE_ORIGIN_HASH = "a" * 40
FAKE_HEAD_HASH = "b" * 40

ORIGIN_XEED = b"#! /usr/bin/env python3\nXEED_ORIGIN_HASH = \"$Format:%H$\"\n"

def _real(filename):
    path = os.path.join(XEED_D_PATH, filename)
    with open(path, "rb") as f:
        return f.read()

ORIGIN_XEED_CFG = lambda: _real("xeed.cfg")

CANONICAL_FILES = {
    "xeed": ORIGIN_XEED.replace(b'"$Format:%H$"', f'"{FAKE_ORIGIN_HASH}"'.encode()),
    "xeed.d/xeed.cfg": lambda: _real("xeed.cfg"),
    "xeed.d/self.cfg": lambda: _real("self.cfg"),
}


def fake_response(data, status=200):
    body = json.dumps(data).encode()
    resp = io.BytesIO(body)
    resp.status = status
    resp.read = resp.read
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    return resp


def make_fake_urlopen(calls=None, branch_exists=False):
    if calls is None:
        calls = []

    def fake_urlopen(request):
        url = request.full_url if hasattr(request, 'full_url') else request
        method = request.method if hasattr(request, 'method') else "GET"
        body = json.loads(request.data) if request.data else None
        calls.append((method, url, body))

        # GET HEAD sha
        if "/git/refs/heads/main" in url:
            return fake_response({"object": {"sha": FAKE_HEAD_HASH}})

        # GET file contents at origin hash
        for path, content in CANONICAL_FILES.items():
            if f"/contents/{path}?" in url and f"ref={FAKE_ORIGIN_HASH}" in url:
                data = content() if callable(content) else content
                return fake_response({
                    "content": base64.b64encode(data).decode(),
                    "sha": "fileshaabc123",
                    "encoding": "base64",
                })

        # POST create branch
        if "/git/refs" in url and method == "POST":
            if branch_exists:
                raise urllib.error.HTTPError(url, 422, "Reference already exists", {}, None)
            return fake_response({"ref": f"refs/heads/self/push-test"})

        # PUT update file
        if "/contents/" in url and method == "PUT":
            return fake_response({"content": {"sha": "newfilesha"}})

        # POST create PR
        if "/pulls" in url and method == "POST":
            return fake_response({"number": 42, "html_url": "https://github.com/brianthelion/xeed/pull/42"})

        raise AssertionError(f"Unexpected API call: {method} {url}")

    return fake_urlopen, calls


@pytest.fixture(scope="module")
def xeed():
    loader = importlib.machinery.SourceFileLoader("xeed", XEED_PATH)
    spec = importlib.util.spec_from_loader("xeed", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture
def project_dir(tmp_path, xeed, monkeypatch):
    import shutil
    xeed_d = tmp_path / "xeed.d"
    xeed_d.mkdir()
    xeed_script = tmp_path / "xeed"
    xeed_script.write_bytes(ORIGIN_XEED.replace(
        b'"$Format:%H$"', f'"{FAKE_ORIGIN_HASH}"'.encode()
    ))
    shutil.copy(os.path.join(XEED_D_PATH, "xeed.cfg"), xeed_d / "xeed.cfg")
    shutil.copy(os.path.join(XEED_D_PATH, "self.cfg"), xeed_d / "self.cfg")
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


# --- self/push tests ---

def test_push_no_changes_is_noop(xeed, project_dir):
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        result = run_cmd(xeed, project_dir, "self/push")
    assert result == 0 or result is None
    methods = [method for method, _, _ in calls]
    assert "POST" not in methods  # no branch created, no PR opened

def test_push_missing_args_exits_with_error(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        result = run_cmd(xeed, project_dir, "self/push")
    assert result != 0
    assert not any(method == "POST" for method, _, _ in calls)

def test_push_missing_message_exits_with_error(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        result = run_cmd(xeed, project_dir, "self/push", "my-branch")
    assert result != 0
    assert not any(method == "POST" for method, _, _ in calls)

def test_push_creates_branch_on_changes(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        result = run_cmd(xeed, project_dir, "self/push", "my-branch", "my commit message")
    assert result == 0 or result is None
    post_urls = [url for method, url, _ in calls if method == "POST"]
    assert any("/git/refs" in url for url in post_urls)

def test_push_opens_pr_on_changes(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        run_cmd(xeed, project_dir, "self/push", "my-branch", "my commit message")
    post_urls = [url for method, url, _ in calls if method == "POST"]
    assert any("/pulls" in url for url in post_urls)

def test_push_excludes_dunder_files(xeed, project_dir):
    (project_dir / "xeed.d" / "__project__.cfg").write_bytes(b"[project]\nname: test\n")
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        run_cmd(xeed, project_dir, "self/push", "my-branch", "my commit message")
    put_urls = [url for method, url, _ in calls if method == "PUT"]
    assert not any("__project__" in url for url in put_urls)

def test_push_custom_branch(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        run_cmd(xeed, project_dir, "self/push", "my-custom-branch", "my commit message")
    branch_bodies = [body for method, url, body in calls if method == "POST" and "/git/refs" in url]
    assert branch_bodies and branch_bodies[0].get("ref") == "refs/heads/my-custom-branch"


def test_push_custom_message(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        run_cmd(xeed, project_dir, "self/push", "my-branch", "my custom message")
    put_bodies = [body for method, url, body in calls if method == "PUT"]
    assert put_bodies and all(b.get("message") == "my custom message" for b in put_bodies)

def test_push_existing_branch_continues(xeed, project_dir):
    (project_dir / "xeed.d" / "xeed.cfg").write_bytes(
        ORIGIN_XEED_CFG() + b"# local change\n"
    )
    fake_urlopen, calls = make_fake_urlopen(branch_exists=True)
    with patch("urllib.request.urlopen", fake_urlopen):
        result = run_cmd(xeed, project_dir, "self/push", "my-branch", "my commit message")
    assert result == 0 or result is None
    put_urls = [url for method, url, _ in calls if method == "PUT"]
    assert put_urls  # files were still pushed
    pr_urls = [url for method, url, _ in calls if method == "POST" and "/pulls" in url]
    assert pr_urls  # PR was still opened
