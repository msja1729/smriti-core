from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer
from smriti.agent.tools import Fixtures, ToolRunner
from smriti.gateway.gateway import Gateway
from smriti.gateway.read_gate import RenderedContext
from smriti.store.fake_store import FakeStore

URL = "https://northwind.example/billing"


def runner(confirm=lambda tool, args: False, tools=("read_web", "send_email", "pay")):
    store = FakeStore()
    gw = Gateway(store, FakeEmbedder(), FakePii(), FakeJudge(), FakeSummarizer())
    return ToolRunner(gw, Fixtures(web={URL: "Northwind page"}), confirm, tools), store


def ctx(tainted, ids=("m1",)):
    return RenderedContext("", "", tainted, list(ids))


def run(r, name, args, tainted=False):
    return r.run(name, args, ctx(tainted), episode_id="ep", session_id="s1", step=2)


def test_untainted_send_executes_and_logs_normalized_args():
    r, store = runner()
    out = run(r, "send_email", {"to": "Billing@Acme.com", "subject": "x", "body": "y"})
    assert out.decision == "executed"
    (logged,) = store.tool_log("ep")
    assert logged.args["to"] == ["billing@acme.com"] and logged.args["cc"] == []
    assert logged.step == 2 and logged.memory_ids_in_context == ["m1"] and not logged.tainted


def test_tainted_risky_denied_by_user():
    r, store = runner(confirm=lambda tool, args: False)
    out = run(r, "send_email", {"to": ["a@evil.io"], "subject": "x", "body": "y"}, tainted=True)
    assert out.decision == "confirm_denied" and "denied" in out.text
    assert store.tool_log("ep")[0].decision == "confirm_denied"
    assert store.tool_log("ep")[0].tainted


def test_tainted_risky_approved_by_user():
    r, store = runner(confirm=lambda tool, args: True)
    out = run(r, "send_email", {"to": ["billing@acme.com"], "subject": "x", "body": "y"},
              tainted=True)
    assert out.decision == "executed"


def test_confirmer_error_counts_as_denial():
    def boom(tool, args):
        raise RuntimeError("confirm crashed")
    r, _ = runner(confirm=boom)
    out = run(r, "pay", {"payee": "x", "account": "y", "amount": 1}, tainted=True)
    assert out.decision == "confirm_denied"


def test_read_web_returns_fixture_or_404():
    r, _ = runner()
    assert run(r, "read_web", {"url": URL}).text == "Northwind page"
    assert "404" in run(r, "read_web", {"url": "https://nope.example"}).text


def test_unknown_tool_rejected():
    r, store = runner()
    out = run(r, "deploy", {"repo": "x", "ref": "main"})     # not in this persona's tools
    assert out.decision == "rejected_fail_closed"
    assert store.tool_log("ep")[0].decision == "rejected_fail_closed"


def test_missing_args_rejected():
    r, _ = runner()
    out = run(r, "send_email", {"subject": "x"})
    assert out.decision == "rejected_fail_closed" and "to" in out.text


def test_schemas_only_for_persona_tools():
    r, _ = runner(tools=("read_web",))
    assert [s["function"]["name"] for s in r.schemas()] == ["read_web"]
