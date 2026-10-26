"""Property: a sealed value is opened only for the tool. It never appears in stored memory,
summaries, the turn log, the ledger, tool_log, or anything the model is sent."""
from hypothesis import given, settings, strategies as st

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer, ScriptedChat
from smriti.agent.graph import Agent
from smriti.agent.tools import Fixtures, ToolRunner
from smriti.gateway.gateway import Gateway, GatewayConfig
from smriti.models.nim import LlmReply, LlmToolCall
from smriti.models.pii import SEALED, PiiSealer
from smriti.store.fake_store import FakeStore

SEALER = PiiSealer(bytes(range(32)))
TENANT = "personal"

phones = st.from_regex(r"\+1 [2-9]\d{2} \d{3} \d{4}", fullmatch=True)
words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ,.", max_size=40)


@settings(max_examples=60, deadline=None)
@given(phones, words, words, st.booleans())
def test_phone_reaches_only_the_tool(phone, before, after, tainted_step):
    store = FakeStore()
    gw = Gateway(store, FakeEmbedder(), FakePii(sealer=SEALER), FakeJudge(), FakeSummarizer(),
                 GatewayConfig(use_judge=False))
    a = gw.write(f"{before} my phone is {phone} {after}", "user", TENANT, "setup").memory_id
    b = gw.write("Dentist check-up on 3 Nov.", "user", TENANT, "setup").memory_id
    gw.consolidate(TENANT, [a, b], "c")
    if tainted_step:
        gw.write("Newsletter about dentist phone forms.", "tool:web", TENANT, "s0")
    summary = [e for e in store.ledger(TENANT) if e.op == "summarize"][0].memory_id
    stored = SEALED.search(store.get(TENANT, summary).text).group(0)

    call = LlmToolCall("c", "fill_form", {"form": "dentist", "fields": {"phone": stored}})
    llm = ScriptedChat([LlmReply(None, (call,)), LlmReply("Done.", ())])
    tools = ToolRunner(gw, Fixtures(), lambda tool, args: True, ("fill_form",))
    Agent(gw, llm, tools, "personal assistant", TENANT).run_turn(
        f"Fill the dentist phone form; my number is {phone}.", "s1", "ep")

    # The tool, and only the tool, got the real number.
    assert tools.delivered == [("fill_form", {"form": "dentist", "fields": {"phone": phone}})]
    everything_else = repr([
        [m.text for m in (store.get(TENANT, mid) for mid in (a, b, summary))],
        [t.user_text for t in store.turns(TENANT, "ep")],
        [e.detail for e in store.ledger(TENANT)],
        [(c.args, c.decision) for c in store.tool_log("ep")],
        # What the model was sent, minus the user's own live request (which the user typed).
        [[m for m in msgs if m["role"] != "user"] for msgs in llm.sent],
    ])
    assert phone not in everything_else
