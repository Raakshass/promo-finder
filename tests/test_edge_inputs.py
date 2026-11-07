"""Brutal input testing: the interview will run this on a show we have never seen. Every variant is generated from the
synthetic fixtures (never the confidential pack) and must give the same promos as the clean baseline, or fail loudly
with a readable message."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from promo_finder.beat_scoring import score_show
from promo_finder.parse import parse_show
from promo_finder.rank import rank_windows

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE.parent / "config.yaml").read_text())


@pytest.fixture(scope="module")
def variants(tmp_path_factory):
    out = tmp_path_factory.mktemp("edge")
    subprocess.run([sys.executable, str(HERE / "make_edge_inputs.py"), str(HERE / "fixtures"), str(out)], check=True, capture_output=True)
    return out


def _clips(folder):
    eps = parse_show(folder)
    ctx = score_show(eps, None, CFG, None)
    fin, _ = rank_windows(ctx, CFG, "heuristic")
    return [e.number for e in eps], [[(s.episode, round(s.start_s, 2), round(s.end_s, 2)) for s in w.segments] for w in fin[:3]]


@pytest.mark.parametrize("name", ["crlf_bom", "hindi", "hhmmss", "comma_decimals", "cp1252", "md_files", "zipped.zip", "one_file_all", "empty_file"])
def test_format_variants_give_the_baseline_promos(variants, name):
    assert _clips(variants / name) == _clips(variants / "baseline")


def test_numbers_come_from_file_names_or_order(variants):
    assert _clips(variants / "no_header")[0] == [7, 8]
    assert _clips(variants / "numbers_101")[0] == [101, 102]


def test_no_camera_blocks_still_gives_valid_clips(variants):
    eps, clips = _clips(variants / "no_camera")
    assert eps == [1, 2] and clips
    for segs in clips:
        assert 20 <= sum(e - s for _, s, e in segs) <= 60


def test_too_short_show_gives_no_clip_instead_of_a_bad_one(variants):
    eps, clips = _clips(variants / "too_short")
    assert eps == [1] and clips == []


def test_garbage_fails_loudly_with_the_expected_format(variants):
    with pytest.raises(ValueError, match="Expected one screenplay per episode"):
        parse_show(variants / "garbage")


def test_gap_in_numbering_never_joins_non_adjacent_episodes(variants):
    _, clips = _clips(variants / "gap_numbers")
    assert all(len({ep for ep, _, _ in segs}) == 1 for segs in clips)


def test_unusual_labels_are_classified_not_guessed(variants):
    eps = parse_show(variants / "weird_labels")
    kinds = {(e.label.split("(")[0].strip(), e.kind, e.mode) for ep in eps for e in ep.events}
    assert ("TEXT ON SCREEN", "other", None) in {(a.rstrip(":"), b, c) for a, b, c in kinds}
    assert any(k[0].startswith("NARRATOR") and k[1] == "dialogue" and k[2] for k in kinds)
