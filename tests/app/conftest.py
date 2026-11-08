"""Fixtures for the app tests.

PROMO_OUTPUTS_DIR is pointed at <repo>/submission/outputs (the six bundle runs) before any app module is imported, so
app.ui.data.OUT resolves there. Tests that need the official run skip when the folder is absent (it is not in git).
The official run is found by its OFFICIAL marker, not by folder name, so a new official run needs no test changes.
Shared helpers live in apphelpers.py; literal expected values for the official run live in expected_official.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apphelpers import APP, BUNDLE, find_official  # noqa: E402

os.environ["PROMO_OUTPUTS_DIR"] = str(BUNDLE)
APP_TIMEOUT_S = 120  # first runs import pandas and plotly; generous for a throttled machine


def pytest_configure(config):
    """Conftests outside packages all import as module "conftest", and pytest leaves the newest in sys.modules.
    This one loads before tests/test_*.py are collected, which do `from conftest import FIX`, so hand the name back to
    tests/conftest.py (pytest calls this hook as soon as this conftest is registered)."""
    parent = config.pluginmanager.get_plugin(str(ROOT / "tests" / "conftest.py"))
    if parent is not None:
        sys.modules["conftest"] = parent


@pytest.fixture(scope="session")
def official_dir() -> Path:
    d = find_official()
    if d is None:
        pytest.skip(f"no official run under {BUNDLE} (the bundle is not in git)")
    return d


@pytest.fixture
def app_test(official_dir, monkeypatch):
    """A fresh, not yet run AppTest of app/streamlit_app.py on the bundle, with no OUTPUTS_URL and no APP_PASSWORD.
    Set query_params or session_state, then call .run()."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("OUTPUTS_URL", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    st.cache_data.clear()
    return AppTest.from_file(str(APP), default_timeout=APP_TIMEOUT_S)
