import pytest

from promo_finder.models import fmt_ts
from promo_finder.parse import parse_episode, parse_label, parse_report

from conftest import FIX


def test_parses_both_fixtures(episodes):
    assert [e.number for e in episodes] == [1, 2]
    e1 = episodes[0]
    assert e1.logline.startswith("Mara is left")
    assert e1.duration == pytest.approx(40.0)
    kinds = {e.kind for e in e1.events}
    assert kinds == {"camera", "dialogue", "action", "sfx", "other"}
    assert e1.blocks_seen == e1.blocks_parsed == 18


def test_uppercase_scene_labels_are_handled(episodes):
    e2 = episodes[1]
    assert e2.scenes[0].heading.startswith("INT. CHAPEL")
    assert e2.scenes[0].t1 == pytest.approx(35.0)
    assert any("UPPERCASE" in w for w in e2.warnings)


def test_dash_normalisation(episodes):
    assert episodes[0].scenes[0].heading == "INT. CHAPEL - DAY"
    assert episodes[1].scenes[0].heading == "INT. CHAPEL - CONTINUOUS"


def test_label_parsing_modes_and_emotions():
    assert parse_label("LENA (O.S.) (Gloating)") == ("dialogue", "LENA", "O.S.", ["Gloating"])
    assert parse_label("FATHER (V.O. / Continues)") == ("dialogue", "FATHER", "V.O.", ["Continues"])
    assert parse_label("LADY ELEANOR (Trembling, Pleading)") == ("dialogue", "LADY ELEANOR", None, ["Trembling", "Pleading"])
    assert parse_label("SOUND FX") == ("sfx", None, None, [])
    assert parse_label("MUSIC")[0] == "other"


def test_overrun_block_is_clamped_with_warning(episodes):
    e1 = episodes[0]
    sfx = [e for e in e1.events if e.kind == "sfx"][0]
    assert sfx.t1 == pytest.approx(40.0)
    assert any("overruns" in w for w in e1.warnings)


def test_dialogue_quotes_stripped(episodes):
    line = [e for e in episodes[0].events if e.kind == "dialogue"][0]
    assert line.text == "Tom?"


def test_parse_rate_guard(tmp_path):
    bad = tmp_path / "episode-009.txt"
    bad.write_text("EPISODE NUMBER: Episode 9\n**CAMERA:**\nno timestamp here\n\n**ACTION:**\nnothing\n")
    with pytest.raises(ValueError):
        parse_episode(bad)


def test_report_and_drift(episodes):
    rep = parse_report(episodes)
    assert rep["episodes"] == 2 and rep["episodes_with_warnings"] == 2
    assert "MARA" in rep["drift"]["speakers"]


def test_fmt_ts():
    assert fmt_ts(0) == "00:00.00"
    assert fmt_ts(83.19) == "01:23.19"


def test_timestamps_with_hours_and_commas():
    from promo_finder.parse import parse_timestamp
    assert parse_timestamp("00:01:23.50") == 83.5
    assert parse_timestamp("01:23,5") == 83.5
    assert parse_timestamp("02:18.00") == 138.0


def test_zip_and_single_file_inputs(tmp_path):
    import shutil
    import zipfile
    from promo_finder.parse import parse_show
    z = tmp_path / "pack.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in FIX.glob("*.txt"):
            zf.write(f, f"show/{f.name}")
    assert [e.number for e in parse_show(z)] == [1, 2]
    one = tmp_path / "ep.txt"
    shutil.copy(FIX / "episode-002-synthetic.txt", one)
    assert [e.number for e in parse_show(one)] == [2]
