import configparser
import importlib.machinery
import importlib.util
import os
import pytest
from unittest.mock import patch

XEED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xeed"))

@pytest.fixture(scope="module")
def xeed():
    loader = importlib.machinery.SourceFileLoader("xeed", XEED_PATH)
    spec = importlib.util.spec_from_loader("xeed", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

def _fake_urlretrieve(fetched):
    def _inner(url, dest):
        fetched[url] = dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(f"# fetched from {url}\n")
    return _inner

FAKE_HASH = "a" * 40

def test_bootstrap_creates_files(xeed, tmp_path):
    fetched = {}
    with patch.object(xeed, "XEED_ORIGIN_HASH", FAKE_HASH):
        with patch("urllib.request.urlretrieve", _fake_urlretrieve(fetched)):
            result = xeed.bootstrap(base_dir=str(tmp_path))
    assert result is True
    assert (tmp_path / "xeed.d" / "xeed.cfg").exists()
    assert (tmp_path / "xeed.d" / "self.cfg").exists()
    assert (tmp_path / "xeed.d" / "__xeed__.cfg").exists()

def test_bootstrap_unexpanded_hash_uses_main(xeed, tmp_path):
    fetched = {}
    with patch.object(xeed, "XEED_ORIGIN_HASH", "$Format:%H$"):
        with patch("urllib.request.urlretrieve", _fake_urlretrieve(fetched)):
            result = xeed.bootstrap(base_dir=str(tmp_path))
    assert result is True
    raw_base = "https://raw.githubusercontent.com/brianthelion/xeed"
    assert any("main" in url for url in fetched)

def test_bootstrap_fetches_correct_urls(xeed, tmp_path):
    fetched = {}
    with patch.object(xeed, "XEED_ORIGIN_HASH", FAKE_HASH):
        with patch.object(xeed, "XEED_ORIGIN_REPO", "https://github.com/brianthelion/xeed"):
            with patch("urllib.request.urlretrieve", _fake_urlretrieve(fetched)):
                xeed.bootstrap(base_dir=str(tmp_path))
    raw_base = "https://raw.githubusercontent.com/brianthelion/xeed"
    assert f"{raw_base}/{FAKE_HASH}/xeed.d/xeed.cfg" in fetched
    assert f"{raw_base}/{FAKE_HASH}/xeed.d/self.cfg" in fetched

def test_bootstrap_install_cfg_content(xeed, tmp_path):
    fetched = {}
    with patch.object(xeed, "XEED_ORIGIN_HASH", FAKE_HASH):
        with patch("urllib.request.urlretrieve", _fake_urlretrieve(fetched)):
            xeed.bootstrap(base_dir=str(tmp_path))
    cfg = configparser.ConfigParser()
    cfg.read(str(tmp_path / "xeed.d" / "__xeed__.cfg"))
    assert cfg["xeed.install"]["XEED_ORIGIN_HASH"] == FAKE_HASH
