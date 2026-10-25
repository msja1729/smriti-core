"""The demo script end to end with fakes (no model, no DB): every step runs."""
import importlib.util
from pathlib import Path

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer, GullibleChat
from smriti.bench.episodes import Components
from smriti.personas import get_persona
from smriti.store.fake_store import FakeStore

ROOT = Path(__file__).resolve().parents[2]


def load_demo():
    spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_runs_all_steps(capsys, tmp_path):
    demo = load_demo()
    parts = Components(GullibleChat(get_persona("finance")), FakeSummarizer(), FakeEmbedder(),
                       FakeJudge(), FakePii())
    demo.main(["--results", str(tmp_path)], parts=parts, make_store=FakeStore)
    out = capsys.readouterr().out
    for n in range(2, 9):
        assert f"== {n}." in out
    assert "attack_succeeded = \033[31mTrue" in out          # A0 attacked
    assert "op=quarantine" in out                             # A2 ledger row
    assert '<memory_data nonce="' in out                      # escaped data block shown
    assert "confirm_denied" in out
    assert "summary trust = \033[32mLOW" in out
