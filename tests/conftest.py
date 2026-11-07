from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


@pytest.fixture(scope="session")
def episodes():
    from promo_finder.parse import parse_show

    return parse_show(FIX)
