"""The traction-learning loop (promo_finder/learn.py): synthetic data and the synthetic fixtures only, no model calls."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from promo_finder import learn as lr
from promo_finder.cli import app

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
W_OLD = np.array([0.35, 0.30, 0.25, 0.10])


def _pool(rng, n=300):
    """Clips with terms shaped like a real run's: hook and cliff on the model's 0-3 scale, standalone and density
    continuous, and the editor's-check multipliers."""
    X = np.column_stack([rng.choice([0, 1 / 3, 2 / 3, 1], n, p=[0.1, 0.2, 0.4, 0.3]),
                         rng.choice([0, 1 / 3, 2 / 3, 1], n, p=[0.1, 0.2, 0.3, 0.4]),
                         rng.uniform(0.2, 1.0, n), rng.uniform(0.3, 1.0, n)])
    m = rng.choice([1.0, 0.92, 0.9, 0.85, 0.828, 0.75, 1.05], n)
    return X * m[:, None]


def _metric(rng, Z, w, r2=0.6, sd=0.03):
    """A hook-rate-like metric whose variance the score under weights `w` explains `r2` of."""
    s = Z @ w
    b = np.sqrt(r2 / (1 - r2)) * sd / s.std()
    return 0.25 + b * (s - s.mean()) + rng.normal(0, sd, len(s))


def test_recovers_known_weights_from_500_posts():
    rng = np.random.default_rng(1)
    pool = _pool(rng)
    w_true = np.array([0.15, 0.45, 0.25, 0.15])
    Z = pool[rng.integers(0, len(pool), 500)]
    f = lr.fit(Z, _metric(rng, Z, w_true), W_OLD, "continuous", 30, pool)
    assert np.abs(f.new - w_true).max() < 0.04, f.new
    assert np.isclose(f.new.sum(), 1) and (f.new >= 0).all()
    assert (f.sd < 0.03).all() and f.data_share > 0.9 and not f.floor
    assert np.all(np.abs(f.new - w_true) <= 3 * f.sd + 0.01)  # the Laplace sd is the right size


def test_recovery_holds_across_seeds():
    """Not one lucky seed: 20 simulations with random true weights. Each weight lands within what the method promises:
    the prior's remaining pull (1 - data share of the distance) plus 3 posterior sd."""
    errs = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        pool = _pool(rng)
        Z = pool[rng.integers(0, len(pool), 500)]
        w_true = rng.dirichlet([4, 4, 4, 4])
        f = lr.fit(Z, _metric(rng, Z, w_true), W_OLD, "continuous", 30, pool)
        assert np.all(np.abs(f.new - w_true) <= (1 - f.data_share) * np.abs(W_OLD - w_true) + 3 * f.sd + 0.01), seed
        errs.append(float(np.abs(f.new - w_true).max()))
    assert np.median(errs) < 0.04, errs


def test_three_posts_stay_near_the_prior():
    rng = np.random.default_rng(2)
    pool = _pool(rng)
    w_far = np.array([0.05, 0.05, 0.10, 0.80])  # the truth is far away; three posts must not get there
    for seed in range(20):
        r = np.random.default_rng(seed)
        Z = pool[r.integers(0, len(pool), 3)]
        f = lr.fit(Z, _metric(r, Z, w_far), W_OLD, "continuous", 30, pool)
        assert np.abs(f.new - W_OLD).max() < 0.08, (seed, f.new)
        assert any("only 3 posts" in w for w in f.warnings)


def test_about_30_posts_move_the_weights_about_halfway():
    """The documented meaning of the default --prior-strength 30, for a metric the score explains well."""
    rng = np.random.default_rng(3)
    pool = _pool(rng)
    w_true = np.array([0.15, 0.45, 0.25, 0.15])
    d = w_true - W_OLD
    moves = []
    for seed in range(40):
        r = np.random.default_rng(1000 + seed)
        Z = pool[r.integers(0, len(pool), 30)]
        f = lr.fit(Z, _metric(r, Z, w_true, r2=0.6), W_OLD, "continuous", 30, pool)
        moves.append(float((f.new - W_OLD) @ d / (d @ d)))
    assert 0.35 < np.median(moves) < 0.65, np.median(moves)


def test_a_noisy_metric_does_no_harm_and_says_so():
    """30 posts of a metric the score barely explains (R2 = 5%) must not make the weights worse on average, must report
    a wide uncertainty, and must mostly carry the 'no reliable change' warning; a clean metric must help."""
    rng = np.random.default_rng(4)
    pool = _pool(rng)
    w_true = np.array([0.15, 0.45, 0.25, 0.15])
    prior_err = 0.5 * np.abs(W_OLD - w_true).sum()
    res = {}
    for r2 in (0.05, 0.6):
        err, sd, warned = [], [], 0
        for seed in range(30):
            r = np.random.default_rng(2000 + seed)
            Z = pool[r.integers(0, len(pool), 30)]
            f = lr.fit(Z, _metric(r, Z, w_true, r2=r2), W_OLD, "continuous", 30, pool)
            err.append(0.5 * np.abs(f.new - w_true).sum())
            sd.append(f.sd.mean())
            warned += any("2 posterior sd" in w for w in f.warnings)
        res[r2] = (np.median(err), np.median(sd), warned)
    assert res[0.05][0] < prior_err + 0.02 and res[0.05][2] >= 20, res
    assert res[0.6][0] < prior_err - 0.04, res
    assert res[0.05][1] > 2 * res[0.6][1], res


def test_binary_outcomes_use_the_logistic_path():
    rng = np.random.default_rng(5)
    pool = _pool(rng)
    w_true = np.array([0.15, 0.45, 0.25, 0.15])
    Z = pool[rng.integers(0, len(pool), 2000)]
    s = Z @ w_true
    y = (rng.random(len(s)) < 1 / (1 + np.exp(-8 * (s - s.mean())))).astype(float)
    f = lr.fit(Z, y, W_OLD, "binary", 30, pool)
    assert f.kind == "binary" and np.isclose(f.new.sum(), 1)
    assert np.abs(f.new - w_true).max() < 0.08, f.new  # 0.02-0.06 over seeds 5-9
    assert np.all(np.abs(f.new - w_true) <= (1 - f.data_share) * np.abs(W_OLD - w_true) + 3 * f.sd + 0.01)
    assert 0.5 * np.abs(f.new - w_true).sum() < 0.5 * np.abs(W_OLD - w_true).sum()  # closer to the truth than before
    assert f.stats["auc_new_score_in_sample"] >= f.stats["auc_old_score"]
    # a perfectly separable handful (all approvals score higher) stays finite and near the prior
    f3 = lr.fit(Z[:4], np.array([0.0, 0.0, 1.0, 1.0])[np.argsort(np.argsort(Z[:4] @ W_OLD))], W_OLD, "binary", 30, pool)
    assert np.isfinite(f3.new).all() and np.abs(f3.new - W_OLD).max() < 0.1


def test_near_certain_binary_outcomes_do_not_inflate_the_prior():
    """A sharp approve/reject (latent R2 = 0.95) gives near-certain outcomes, and each one teaches little. The prior must
    count as kappa posts of that information, not of a base-rate post's, so 100 such posts still move the weights well
    past halfway (before the fix: 0.13 of the way at n = 30) and the 2-sd band keeps covering the truth."""
    rng = np.random.default_rng(8)
    pool = _pool(rng)
    w_true = np.array([0.15, 0.45, 0.25, 0.15])
    d = w_true - W_OLD
    s_pool = pool @ w_true
    b = np.sqrt(0.95 / 0.05) * (np.pi / np.sqrt(3)) / s_pool.std()
    moves, cover = [], []
    for seed in range(15):
        r = np.random.default_rng(3000 + seed)
        Z = pool[r.integers(0, len(pool), 100)]
        y = (r.random(100) < 1 / (1 + np.exp(-b * (Z @ w_true - s_pool.mean())))).astype(float)
        f = lr.fit(Z, y, W_OLD, "binary", 30, pool)
        moves.append(float((f.new - W_OLD) @ d / (d @ d)))
        cover.append(float(np.mean(np.abs(f.new - w_true) <= 2 * f.sd)))
    assert np.median(moves) > 0.55 and np.mean(cover) > 0.75, (np.median(moves), np.mean(cover))


def test_outcome_files_detect_metric_kind_and_ids(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("promo_id,episode,hook_rate,notes\nP01,1,31%,x\nP02,4,0.25,y\nP03,9,,z\n")
    o = lr.read_outcomes(p)
    assert o.metric == "hook_rate" and o.kind == "continuous" and o.ids == ["P01", "P02"] and o.skipped == 1
    assert o.y.tolist() == [31.0, 0.25]
    p.write_text("rank,approved\n1,yes\n2,0\n3,approved\n4,trim\n")
    o = lr.read_outcomes(p)
    assert o.kind == "binary" and o.id_column == "rank" and o.y.tolist() == [1.0, 0.0, 1.0] and o.skipped == 1
    p.write_text("promo_id,hold_rate,ctr\nP01,0.5,0.1\n")
    assert lr.read_outcomes(p).metric == "hold_rate"
    assert lr.read_outcomes(p, metric="ctr").metric == "ctr"
    p.write_text("clip,views\nP01,3\n")
    with pytest.raises(lr.LearnError, match="promo_id column"):
        lr.read_outcomes(p)
    d = tmp_path / "decisions.json"  # the app's approve/reject log
    d.write_text(json.dumps({"P01": "approved", "P02": "rejected", "P03": "trim"}))
    o = lr.read_outcomes(d)
    assert o.kind == "binary" and o.ids == ["P01", "P02"] and o.y.tolist() == [1.0, 0.0]


def test_no_variation_or_too_few_posts_is_a_readable_error():
    with pytest.raises(lr.LearnError, match="same outcome"):
        lr.fit(np.ones((5, 4)) * 0.5, np.ones(5), W_OLD)
    with pytest.raises(lr.LearnError, match="at least 2"):
        lr.fit(np.ones((1, 4)), np.ones(1), W_OLD)


# ------------------------------------------------------------------------------------- a real run on the fixtures
runner = CliRunner()


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A heuristic run of the synthetic fixtures. The run command must not read the developer's .env (it would leave
    real keys in os.environ for later tests), so load_dotenv is disabled and the environment restored."""
    import os
    import promo_finder.cli as cli
    out = tmp_path_factory.mktemp("runs")
    env = dict(os.environ)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "load_dotenv", lambda *a, **k: False)
        r = runner.invoke(app, ["run", str(FIX), "--model", "none", "--out", str(out)])
    os.environ.clear()
    os.environ.update(env)
    assert r.exit_code == 0, r.output
    return next(out.glob("*/*"))


def test_pool_joins_promos_and_candidates_with_their_terms(run_dir):
    pool = lr.load_pool(run_dir)
    promos = json.loads((run_dir / "promos.json").read_text())
    assert len(pool.clips) >= len(promos) and np.allclose(pool.weights, W_OLD)
    for c in pool.clips:  # every clip's stored score is mult x terms . w_old (rounding only)
        assert abs(c.mult * c.terms @ pool.weights - c.score) < 2e-3
    assert pool.find("P01").promo_id == "P01" and pool.find("1") is pool.find("p1")
    assert pool.find(promos[0]["flags"]["cand_id"]) is pool.find("P01")
    assert pool.find("P99") is None and pool.find("e999-001") is None


def test_older_runs_rebuild_candidate_terms_from_their_own_files(run_dir, tmp_path):
    """Runs from before candidates.jsonl stored the terms: they are rebuilt (no model calls) and must match."""
    old = tmp_path / "old_run"
    old.mkdir()
    for f in run_dir.iterdir():
        if f.is_file():
            (old / f.name).write_bytes(f.read_bytes())
    (old / "beats").mkdir()
    for f in (run_dir / "beats").iterdir():
        (old / "beats" / f.name).write_bytes(f.read_bytes())
    keep = ("cand_id", "score", "block", "kind", "segments")
    rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines()]
    (old / "candidates.jsonl").write_text("".join(json.dumps({k: r[k] for k in keep}) + "\n" for r in rows))
    # keep only P01 in promos.json, so the other candidates can only get their terms by the rebuild
    (old / "promos.json").write_text(json.dumps(json.loads((run_dir / "promos.json").read_text())[:1]))
    new_pool, old_pool = lr.load_pool(run_dir), lr.load_pool(old)
    rebuilt = [c for c in old_pool.clips if c.source == "replay"]
    assert len(rebuilt) == len(old_pool.clips) - 1 >= 1 and any("rebuilt" in n for n in old_pool.notes)
    for c in rebuilt:
        ref = new_pool.find(c.cand_id)
        assert np.allclose(c.terms, ref.terms) and abs(c.mult - ref.mult) < 1e-9 and c.block == ref.block
    # without the show folder an old run still loads its promos, and says what it had to leave out
    log = json.loads((old / "run_log.json").read_text())
    log["args"]["show_dir"] = str(tmp_path / "moved-away")
    (old / "run_log.json").write_text(json.dumps(log))
    p = lr.load_pool(old)
    assert [c.source for c in p.clips] == ["promos"] and any("--show-dir" in n for n in p.notes)
    assert [c.source for c in lr.load_pool(old, show_dir=FIX).clips].count("replay") == len(rebuilt)


def test_unknown_ids_are_reported_and_skipped(run_dir, tmp_path):
    pool = lr.load_pool(run_dir)
    p = tmp_path / "o.csv"
    p.write_text("promo_id,hook_rate\nP01,0.30\nP99,0.10\nnot-a-clip,0.2\nP02,0.22\n")
    j = lr.join(lr.read_outcomes(p), pool)
    assert [c.promo_id for c in j.clips] == ["P01", "P02"] and j.unknown == ["P99", "not-a-clip"]
    p.write_text("promo_id,hook_rate\nP98,0.30\nP99,0.10\n")
    with pytest.raises(lr.LearnError, match="none of the 2 ids"):
        lr.join(lr.read_outcomes(p), pool)
    r = runner.invoke(app, ["learn", str(run_dir), "--outcomes", str(p)])
    assert r.exit_code == 1 and "none of the 2 ids" in r.output


def test_cli_learn_writes_a_new_config_and_a_plan(run_dir, tmp_path):
    p = tmp_path / "outcomes.csv"
    p.write_text("promo_id,hook_rate,hold_rate\nP01,0.31,0.4\nP02,0.18,0.3\nunknown-id,0.5,0.5\n")
    out = tmp_path / "learned.yaml"
    r = runner.invoke(app, ["learn", str(run_dir), "--outcomes", str(p), "--out", str(out), "--plan", "2", "--json"])
    assert r.exit_code == 0, r.output
    res = json.loads(r.output)
    assert res["metric"] == "hook_rate" and res["fit"]["n"] == 2 and res["unknown_ids"] == ["unknown-id"]
    assert any("only 2 posts" in w for w in res["fit"]["warnings"])
    assert abs(sum(res["fit"]["new"].values()) - 1) < 1e-6
    text = out.read_text()
    assert text.startswith("# Learned ranking weights") and "worth 30 posts" in text
    cfg, base = yaml.safe_load(text), yaml.safe_load((ROOT / "config.yaml").read_text())
    assert abs(sum(cfg["scoring"][k] for k in lr.WEIGHT_KEYS) - 1) < 1e-9 and all(cfg["scoring"][k] >= 0 for k in lr.WEIGHT_KEYS)
    assert {k: v for k, v in cfg.items() if k != "scoring"} == {k: v for k, v in base.items() if k != "scoring"}
    base_comments = [ln for ln in (ROOT / "config.yaml").read_text().splitlines() if ln.strip().startswith("#")]
    assert all(ln in text for ln in base_comments)  # the base file's comments survive
    plan = res["plan"]
    posted = {json.loads((run_dir / "promos.json").read_text())[i]["flags"]["cand_id"] for i in (0, 1)}
    assert plan["k"] == 2 and len(plan["rows"]) <= 2 and not posted & {row["cand_id"] for row in plan["rows"]}
    for row in plan["rows"]:
        assert row["pick"] in ("exploit", "explore") and 0 <= row["in_slate_share"] <= 1
    r2 = runner.invoke(app, ["learn", str(run_dir), "--outcomes", str(p), "--out", str(out), "--plan", "2", "--json"])
    assert json.loads(r2.output)["plan"] == plan  # fixed seed: the same plan


def test_cli_never_overwrites_a_config_yaml(run_dir, tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("promo_id,hook_rate\nP01,0.31\nP02,0.18\n")
    before = (run_dir / "config.yaml").read_text()
    r = runner.invoke(app, ["learn", str(run_dir), "--outcomes", str(p), "--out", str(run_dir / "config.yaml")])
    assert r.exit_code == 1 and "config.yaml" in r.output
    assert (run_dir / "config.yaml").read_text() == before


def test_thompson_plan_explores_and_respects_the_slate_rules():
    """A synthetic pool: the plan is deterministic under its seed, never repeats a posted clip, never picks two
    near-duplicates, and labels picks that the posterior-mode slate would also post as 'exploit'."""
    rng = np.random.default_rng(6)
    clips = []
    for i in range(60):
        ep = 1 + i // 2
        t0 = 10.0 if i % 2 == 0 else 70.0
        clips.append(lr.Clip(cand_id=f"c{i:02d}", terms=rng.uniform(0.3, 1, 4), mult=float(rng.choice([1.0, 0.9, 0.85])), score=0.0,
                             segments=[(ep, t0, t0 + 30)], block="opening" if ep <= 6 else ("middle" if ep <= 15 else "late"),
                             hook_type=str(rng.choice(["slap", "reveal", "threat"]))))
    for c in clips:
        c.score = round(float(c.mult * c.terms @ W_OLD), 4)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pool = lr.Pool(run_dir=Path("."), cfg=cfg, weights=W_OLD, clips=clips)
    posted = clips[:8]
    Z = np.array([c.z for c in posted])
    f = lr.fit(Z, _metric(rng, Z, np.array([0.1, 0.5, 0.3, 0.1]), r2=0.5), W_OLD, "continuous", 30, pool.Z)
    a = lr.plan(f, pool, 5, posted, draws=300, seed=11)
    b = lr.plan(f, pool, 5, posted, draws=300, seed=11)
    assert a == b and len(a["rows"]) == 5 and a["valid_draws"] == 300
    picked = [r["cand_id"] for r in a["rows"]]
    assert not set(picked) & {c.cand_id for c in posted} and len(set(picked)) == 5
    assert sum(r["block"] == "opening" for r in a["rows"]) == 3  # quotas of 5: 3 opening, 2 middle
    for r in a["rows"]:
        assert (r["pick"] == "exploit") == (r["cand_id"] in a["map_slate"])
