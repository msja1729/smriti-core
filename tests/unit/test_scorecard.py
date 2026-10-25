import pytest

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer
from smriti.bench.run import BenchConfig, run
from smriti.scorecard.report import load, render, stop_breakdown, targets, wilson
from smriti.store.fake_store import FakeStore
from test_bench import CFG, parts


def test_wilson_known_values():
    lo, hi = wilson(0, 60)
    assert lo == 0.0 and hi == pytest.approx(0.0602, abs=1e-3)
    lo, hi = wilson(29, 60)
    assert lo == pytest.approx(0.36, abs=0.01) and hi == pytest.approx(0.61, abs=0.01)
    assert wilson(0, 0) is None


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    cfg = BenchConfig(**{**CFG, "ablations": ["no_write_flagging", "no_action_check"]})
    run(cfg, parts(), FakeStore, out_dir=d, log=lambda *_: None)
    return d


def test_render_has_targets_and_tables(run_dir):
    metrics, results = load(run_dir)
    md = render(metrics, results, "t")
    assert "A2 against the targets" in md and "Wilson" in md
    assert "A2 SMRITI" in md and "A2 - write flagging (detector off)" in md
    names = [t[0] for t in targets(metrics)]
    assert "Attack success rate" in names and "Added latency vs A0, p95 (own)" in names


def test_stop_breakdown_counts_every_attack_episode(run_dir):
    metrics, results = load(run_dir)
    b = stop_breakdown(results, "A2")
    assert sum(b.values()) == metrics["asr"]["A2"]["all"]["n"]
    assert b["attack succeeded"] == 0


def test_charts_written(run_dir):
    from smriti.scorecard.charts import write_charts
    metrics, _ = load(run_dir)
    paths = write_charts(metrics, run_dir)
    assert {p.name for p in paths} == {"asr_by_arm_class.png", "utility_by_arm.png",
                                       "ablations.png"}
    assert all(p.stat().st_size > 10_000 for p in paths)
