import importlib.machinery
import importlib.util
import os
import shutil
import sys
import pytest
from unittest.mock import patch

XEED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed"))
XEED_D_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed.d"))

@pytest.fixture(scope="module")
def xeed():
    loader = importlib.machinery.SourceFileLoader("xeed", XEED_PATH)
    spec = importlib.util.spec_from_loader("xeed", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

@pytest.fixture
def config_dir(tmp_path):
    xeed_d = tmp_path / "xeed.d"
    xeed_d.mkdir()
    shutil.copy(os.path.join(XEED_D_PATH, "xeed.cfg"), xeed_d / "xeed.cfg")
    shutil.copy(os.path.join(XEED_D_PATH, "self.cfg"), xeed_d / "self.cfg")
    return tmp_path

@pytest.fixture(autouse=True)
def isolated_pwd(xeed, config_dir):
    orig = xeed.ENV["PWD"]
    xeed.ENV["PWD"] = str(config_dir)
    yield
    xeed.ENV["PWD"] = orig

def run_self_install(xeed, config_dir, tool):
    fetched = {}

    def fake_urlretrieve(url, dest):
        fetched[url] = dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(f"# fetched from {url}\n")

    orig_argv = sys.argv
    try:
        sys.argv = [XEED_PATH, "--config", str(config_dir / "xeed.d"), "self/install", tool]
        xeed.Cli.OPTIONS["--config"] = None
        with patch("urllib.request.urlretrieve", fake_urlretrieve):
            result = xeed.main()
    finally:
        sys.argv = orig_argv

    return result, fetched

def test_self_install_fetches_tool(xeed, config_dir):
    result, fetched = run_self_install(xeed, config_dir, "docker")
    assert result == 0 or result is None
    assert len(fetched) == 1
    url = list(fetched.keys())[0]
    assert url.endswith("/xeed.d/docker.cfg")
    assert "raw.githubusercontent.com" in url

def test_self_install_dest_path(xeed, config_dir):
    _, fetched = run_self_install(xeed, config_dir, "docker")
    dest = list(fetched.values())[0]
    assert dest == str(config_dir / "xeed.d" / "docker.cfg")
