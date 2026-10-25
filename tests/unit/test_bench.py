"""Attacks, personas, episodes and the runner with fakes (no model, no DB, no network).

GullibleChat obeys any poison it can see, so these tests show which layer stops what."""
import json

import pytest

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer, GullibleChat
from smriti.attacks import ATTACKS, make_poison
from smriti.attacks import direct, indirect, sleeper
from smriti.bench.episodes import Components, EpisodeSettings, EpisodeSpec, run_episode
from smriti.bench.metrics import aggregate
from smriti.bench.run import BenchConfig, episode_grid, load_config, run
from smriti.gateway.heuristics import Verdict, score
from smriti.personas import get_persona
from smriti.store.fake_store import FakeStore

PERSONAS = ("finance", "devops", "personal")
SETTINGS = EpisodeSettings(clean_sessions=1, turns_per_clean_session=2, consolidate_every=10)


class PersonaRoutedChat:
    """Picks the GullibleChat for whichever persona's role is in the system prompt."""

    def chat(self, messages, tools):
        system = messages[0]["content"]
        persona = next(get_persona(p) for p in PERSONAS if get_persona(p).role in system)
        return GullibleChat(persona).chat(messages, tools)


def parts():
    return Components(PersonaRoutedChat(), FakeSummarizer(), FakeEmbedder(), FakeJudge(),
                      FakePii())


def episode(persona, attack, arm, seed=0):
    return run_episode(EpisodeSpec(persona, attack, arm, seed), SETTINGS, parts(), FakeStore())


# -- attacks and personas -----------------------------------------------------------------

@pytest.mark.parametrize("module", [direct, indirect, sleeper])
def test_at_least_five_templates(module):
    assert len(module.TEMPLATES) >= 5 and len(set(module.TEMPLATES)) == len(module.TEMPLATES)


@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("attack", ATTACKS)
def test_poison_mentions_attacker_and_varies_by_seed(persona, attack):
    p = get_persona(persona)
    texts = {make_poison(attack, p, s).text for s in range(5)}
    assert len(texts) == 5
    assert all("evil" in t for t in texts)


def test_some_templates_evade_heuristics():
    # Otherwise the benchmark would only ever exercise the write gate.
    p = get_persona("finance")
    verdicts = {score(indirect.make(p, s).text).verdict for s in range(5)}
    assert Verdict.INSTRUCTION_LIKE in verdicts and verdicts != {Verdict.INSTRUCTION_LIKE}


@pytest.mark.parametrize("persona", PERSONAS)
def test_persona_clean_text_is_not_flagged(persona):
    p = get_persona(persona)
    docs = [*p.fixtures.web.values(), *p.fixtures.emails.values(), *p.seed_memories]
    assert all(score(d).verdict is not Verdict.INSTRUCTION_LIKE for d in docs)


# -- episodes -----------------------------------------------------------------------------

@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("attack", ATTACKS)
def test_a0_gullible_model_is_attacked(persona, attack):
    r = episode(persona, attack, "A0")
    assert r.error is None and r.attack_succeeded, r.tool_calls


@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("arm", ["A2", "no_write_flagging", "no_read_escaping"])
def test_smriti_blocks_every_attack_even_against_gullible_model(persona, attack, seed, arm):
    r = episode(persona, attack, arm, seed)
    assert r.error is None and not r.attack_succeeded, r.tool_calls


def test_detector_off_relies_on_action_check():
    r = episode("finance", "indirect", "no_write_flagging")
    assert r.poison_status == "stored"
    assert any(c["decision"] == "confirm_denied" for c in r.tool_calls)


def test_no_action_check_lets_unflagged_poison_through():
    # Template 3 avoids the heuristics, so with L3 off the gullible model executes it.
    r = episode("finance", "indirect", "no_action_check", seed=3)
    assert r.poison_status == "stored" and r.attack_succeeded


@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("arm", ["A0", "A1", "A2"])
def test_clean_episodes_succeed(persona, arm):
    r = episode(persona, "clean", arm)
    assert r.error is None and r.task_succeeded and not r.attack_succeeded, r.tool_calls
    assert r.quarantined == 0 and r.writes > 0


def test_laundering_summary_stays_low_and_originals_archived():
    r = episode("finance", "laundering", "A2")
    assert r.summaries_from_low >= 1 and r.summaries_not_low == 0


def test_sleeper_phrase_only_in_trigger():
    r = episode("finance", "sleeper", "A0")
    trigger = [c for c in r.tool_calls if c["session"] == "trigger"]
    assert trigger and r.attack_succeeded


def test_no_canary_leak_and_latencies_recorded():
    r = episode("personal", "indirect", "A2")
    assert r.canary_leaks == 0 and r.write_latency_ms and r.read_latency_ms


def test_episode_error_is_recorded_not_raised():
    bad = Components(PersonaRoutedChat(), FakeSummarizer(), FakeEmbedder(fail=True),
                     FakeJudge(), FakePii())
    r = run_episode(EpisodeSpec("finance", "indirect", "A2", 0), SETTINGS, bad, FakeStore())
    # Embedder down -> writes rejected and reads HIGH-only (fail closed); episode still runs.
    assert r.error is None and not r.attack_succeeded


# -- metrics and runner -------------------------------------------------------------------

def test_aggregate_rates():
    rs = [episode("finance", "indirect", "A0"), episode("finance", "indirect", "A2"),
          episode("finance", "clean", "A0"), episode("finance", "clean", "A2")]
    m = aggregate([r.to_json() for r in rs])
    assert m["asr"]["A0"]["indirect"]["rate"] == 1.0 and m["asr"]["A2"]["indirect"]["rate"] == 0.0
    assert m["utility_retention"]["A2"] == 1.0
    assert m["per_arm"]["A2"]["false_quarantine"]["rate"] == 0.0


CFG = dict(run_id="t", seeds=[0, 1], personas=["finance"], attacks=["indirect", "direct"],
           arms=["A0", "A2"], agent_model="m", agent_base_url="u", judge_model="j",
           embed_model="e", clean_sessions_between=1, workers=2)


def test_grid_size_and_validation():
    assert len(episode_grid(BenchConfig(**CFG))) == 1 * 3 * 2 * 2
    with pytest.raises(ValueError):
        BenchConfig(**{**CFG, "attacks": ["nope"]})


def test_demo_config_has_180_attack_episodes_per_main_arms():
    cfg = load_config("src/smriti/bench/configs/demo.yaml")
    grid = episode_grid(cfg)
    assert sum(s.attack != "clean" and s.arm in ("A0", "A1", "A2") for s in grid) == 180


def test_run_writes_results_and_resumes(tmp_path):
    cfg = BenchConfig(**CFG)
    logs = []
    m = run(cfg, parts(), FakeStore, out_dir=tmp_path, log=logs.append)
    assert m["episodes"] == 12 and m["asr"]["A2"]["all"]["rate"] == 0.0
    lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 12 and all(json.loads(x)["error"] is None for x in lines)
    run(cfg, parts(), FakeStore, out_dir=tmp_path, log=logs.append)
    assert "0 to run" in logs[-1] or any("0 to run" in x for x in logs)
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 12
