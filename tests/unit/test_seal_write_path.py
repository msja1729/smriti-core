"""Sealed PII on the write path: memories, consolidation, turn log and prompts (no model)."""
from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer, ScriptedChat
from smriti.agent.graph import Agent
from smriti.agent.prompts import SEALED_NOTE, system_prompt
from smriti.agent.tools import Fixtures, ToolRunner
from smriti.gateway.gateway import Gateway, GatewayConfig
from smriti.gateway.labels import Sensitivity
from smriti.gateway.read_gate import RenderedContext
from smriti.models.nim import SUMMARY_SEALED_NOTE, LlmReply, NimSummarizer
from smriti.models.pii import SEALED, PiiSealer
from smriti.store.fake_store import FakeStore

SEALER = PiiSealer(bytes(range(32)))
PHONE = "+1 415 555 0132"


def make(sealer=SEALER, replies=()):
    store = FakeStore()
    gw = Gateway(store, FakeEmbedder(), FakePii(sealer=sealer), FakeJudge(), FakeSummarizer(),
                 GatewayConfig(use_judge=False))
    llm = ScriptedChat(list(replies))
    agent = Agent(gw, llm, ToolRunner(gw, Fixtures(), lambda t, a: False, ("send_email",)),
                  role="personal assistant", tenant="personal")
    return gw, store, agent, llm


def test_user_memory_is_sealed_not_lost():
    gw, store, _, _ = make()
    r = gw.write(f"My phone number is {PHONE}.", "user", "personal", "setup")
    m = store.get("personal", r.memory_id)
    assert "555" not in m.text and SEALED.search(m.text)
    assert m.label.sensitivity == Sensitivity.PII
    assert SEALER.unseal(m.text, "personal").text == f"My phone number is {PHONE}."


def test_sealed_under_the_memorys_own_tenant():
    gw, store, _, _ = make()
    r = gw.write(f"Call {PHONE}.", "user", "personal", "setup")
    text = store.get("personal", r.memory_id).text
    assert SEALER.unseal(text, "personal").text == f"Call {PHONE}."


def test_no_sealer_keeps_permanent_redaction():
    gw, store, _, _ = make(sealer=None)
    r = gw.write(f"My phone number is {PHONE}.", "user", "personal", "setup")
    assert store.get("personal", r.memory_id).text == "My phone number is [PHONE]."


def test_ledger_never_holds_the_value_or_ciphertext():
    gw, store, _, _ = make()
    gw.write(f"My phone number is {PHONE}.", "user", "personal", "setup")
    dump = repr([e.detail for e in store.ledger("personal")])
    assert "555" not in dump and not SEALED.search(dump)


def test_consolidation_keeps_sealed_value_intact():
    gw, store, _, _ = make()
    a = gw.write(f"My phone number is {PHONE}.", "user", "personal", "s1").memory_id
    b = gw.write("Dinner with Priya on Friday.", "user", "personal", "s2").memory_id
    summary = gw.consolidate("personal", [a, b], "s3")
    assert "555" not in summary.text
    assert PHONE in SEALER.unseal(summary.text, "personal").text


def test_turn_log_is_sealed():
    _, store, agent, _ = make(replies=[LlmReply("ok", ())])
    agent.run_turn(f"Remember my phone is {PHONE}.", "s1", "ep")
    (turn,) = store.turns("personal", "ep")
    assert "555" not in turn.user_text
    assert PHONE in SEALER.unseal(turn.user_text, "personal").text


def test_prompt_note_only_when_memory_holds_sealed_values():
    sealed = SEALER.seal(PHONE, "PHONE_NUMBER", "t")
    plain = system_prompt("assistant", RenderedContext("- My phone is [PHONE].", "", False, []))
    assert SEALED_NOTE not in plain
    for ctx in (RenderedContext(f"- My phone is {sealed}.", "", False, []),
                RenderedContext("", f"<memory_data>{sealed}</memory_data>", True, [])):
        assert SEALED_NOTE in system_prompt("assistant", ctx)


def test_agent_sees_note_and_sealed_value_not_the_phone():
    gw, _, agent, llm = make(replies=[LlmReply("ok", ())])
    gw.write(f"My phone number is {PHONE}.", "user", "personal", "setup")
    agent.run_turn("What is my phone number?", "s1", "ep")
    system = llm.sent[0][0]["content"]
    assert SEALED_NOTE in system and SEALED.search(system) and "555" not in system


def test_summary_prompt_note_only_with_sealed_notes():
    sealed = SEALER.seal(PHONE, "PHONE_NUMBER", "t")
    for notes, expect in ([["a", "b"], False], [[f"call {sealed}", "b"], True]):
        llm = ScriptedChat([LlmReply("summary", ())])
        NimSummarizer(llm).summarize(notes)
        assert (SUMMARY_SEALED_NOTE in llm.sent[0][0]["content"]) is expect
