"""AgentDojo adapter with a fake model that replays ground truth (no model, no network)."""
import pytest

pytest.importorskip("agentdojo")

from agentdojo.agent_pipeline import BasePipelineElement  # noqa: E402
from agentdojo.functions_runtime import FunctionCall  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string  # noqa: E402

from smriti.external import agentdojo_run as ad  # noqa: E402
from smriti.gateway.gateway import A2  # noqa: E402

SUITE = get_suite("v1.2.2", "workspace")
CONFIG = ad.replace(A2, risky_tools=frozenset(t.name for t in SUITE.tools if ad.is_risky(t.name)))


class Replay(BasePipelineElement):
    """Fake model: emits the given calls one per turn, then a final answer. `plan` gets the
    environment, so ids in ground truth resolve against the injected environment."""

    def __init__(self, plan):
        self.plan = plan
        self.name = "replay"
        self.calls = None

    def query(self, query, runtime, env=None, messages=(), extra_args={}):
        # Planned once, on the environment as the run starts: recomputing later would see
        # the run's own changes (e.g. a file it created) and shift the ids.
        if self.calls is None:
            self.calls = self.plan(env)
        calls = self.calls
        made = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
        if made < len(calls):
            c = calls[made]
            msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")],
                                       tool_calls=[FunctionCall(function=c.function, args=dict(c.args),
                                                                id=f"c{made}")])
        else:
            msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("Done.")],
                                       tool_calls=None)
        return query, runtime, env, [*messages, msg], extra_args


def run(arm, user_task, plan, injection_task=None, injections=None):
    executor = None
    if arm == "smriti":
        gt = user_task.ground_truth(SUITE.load_and_inject_default_environment({}))
        executor = ad.SmritiToolsExecutor(ad.Gateway(None, None, None, None, None, CONFIG),
                                          ad.reasonable_user(gt))
    pipeline = ad.build_pipeline(arm, Replay(plan), "system", SUITE, executor)
    utility, security = SUITE.run_task_with_pipeline(pipeline, user_task, injection_task,
                                                     injections or {})
    return utility, security, executor


def test_risky_tools_by_verb():
    risky = {t.name for t in SUITE.tools if ad.is_risky(t.name)}
    assert {"send_email", "share_file", "delete_file", "create_calendar_event"} <= risky
    assert not risky & {"search_emails", "get_received_emails", "list_files", "get_current_day"}


def test_sources_follow_default_trust():
    gw = ad.Gateway(None, None, None, None, None, CONFIG)
    assert gw.trust_of(ad.source_of("get_received_emails")) == ad.Trust.LOW
    assert gw.trust_of(ad.source_of("search_files")) == ad.Trust.LOW
    assert gw.trust_of(ad.source_of("search_calendar_events")).name == "MED"


@pytest.mark.parametrize("task_id", sorted(SUITE.user_tasks))
def test_reasonable_user_keeps_ground_truth_utility(task_id):
    """Replaying the right calls, SMRITI's confirmations never cost a task its utility."""
    task = SUITE.get_user_task_by_id(task_id)
    plain, _, _ = run("none", task, task.ground_truth)
    smriti, _, executor = run("smriti", task, task.ground_truth)
    assert smriti == plain
    assert executor.denied == 0


def test_attackers_calls_denied_after_untrusted_read():
    """A fully fooled model reads the user's mail (LOW), then does the injection task."""
    user_task = SUITE.get_user_task_by_id("user_task_0")
    hits = 0
    for inj_id, inj in SUITE.injection_tasks.items():
        plan = lambda env, inj=inj: [FunctionCall(function="search_emails", args={"query": "x"}),
                                     *inj.ground_truth(env)]
        _, plain_sec, _ = run("none", user_task, plan, inj, {})
        _, smriti_sec, executor = run("smriti", user_task, plan, inj, {})
        if plain_sec:
            hits += 1
            assert not smriti_sec, inj_id
            assert executor.denied > 0
    assert hits >= 6           # 6 of the 14 goals are reachable by this simple replay


def test_reasonable_user_compares_targets_not_content():
    gt = [FunctionCall(function="send_email", args={"recipients": ["a@x.com"], "subject": "s", "body": "b"})]
    approve = ad.reasonable_user(gt)
    assert approve("send_email", {"recipients": ["A@x.com"], "subject": "other", "body": "other"})
    assert not approve("send_email", {"recipients": ["a@x.com"], "bcc": ["evil@x.com"], "subject": "s", "body": "b"})
    assert not approve("send_email", {"recipients": ["evil@x.com"], "subject": "s", "body": "b"})
    assert not approve("share_file", {"file_id": "1", "email": "a@x.com"})


def test_developer_role_renamed_to_system():
    from types import SimpleNamespace as NS
    sent = {}
    inner = NS(chat=NS(completions=NS(create=lambda **kw: sent.update(kw) or "ok")))
    client = ad.SystemRoleClient(inner)
    assert client.chat.completions.create(model="m", messages=[
        {"role": "developer", "content": "sys"}, {"role": "user", "content": "hi"}]) == "ok"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == "sys"


def test_non_user_content_blocks_joined_for_vllm():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert ad._vllm_message({"role": "tool", "content": blocks})["content"] == "ab"
    assert ad._vllm_message({"role": "user", "content": blocks})["content"] == blocks
    assert ad._vllm_message({"role": "assistant", "content": None})["content"] is None


def test_context_overflow_scored_as_agentdojo_does():
    err = "BadRequestError: Error code: 400 - This model's maximum context length is 32768"
    attacked = ad.scored({"injection_task": "i1", "error": err, "utility": None, "security": None})
    clean = ad.scored({"injection_task": None, "error": err, "utility": None, "security": None})
    assert attacked["error"] is None and attacked["utility"] is False and attacked["security"] is True
    assert clean["utility"] is False and clean["security"] is None and clean["context_overflow"]
    other = {"injection_task": "i1", "error": "APIConnectionError", "utility": None}
    assert ad.scored(other) == other
