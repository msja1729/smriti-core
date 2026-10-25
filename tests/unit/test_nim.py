"""NimChat parsing and retry, against a fake OpenAI client (no network)."""
from types import SimpleNamespace as NS

import pytest

from smriti.models.nim import NimChat, NimConfig, NimSummarizer, ToolCallError

TOOLS = [{"type": "function", "function": {"name": "send_email", "parameters": {}}}]
CFG = NimConfig(base_url="http://x/v1", model="m")


def reply(content=None, calls=()):
    tcs = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tcs or None))])


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.replies.pop(0)


def test_parses_tool_call_and_adds_no_think():
    client = FakeClient([reply(calls=[("send_email", '{"to": ["a@b.c"]}')])])
    out = NimChat(CFG, client).chat([{"role": "system", "content": "sys"},
                                     {"role": "user", "content": "go"}], TOOLS)
    assert out.tool_calls[0].name == "send_email" and out.tool_calls[0].args == {"to": ["a@b.c"]}
    req = client.requests[0]
    assert req["messages"][0]["content"].startswith("/no_think\n")
    assert req["temperature"] == 0.0 and req["model"] == "m"


def test_invalid_json_retried_once():
    client = FakeClient([reply(calls=[("send_email", "{not json")]),
                         reply(calls=[("send_email", "{}")])])
    out = NimChat(CFG, client).chat([{"role": "user", "content": "go"}], TOOLS)
    assert out.tool_calls[0].args == {}
    assert "invalid" in client.requests[1]["messages"][-1]["content"]


def test_unknown_tool_twice_raises():
    client = FakeClient([reply(calls=[("rm_rf", "{}")]), reply(calls=[("rm_rf", "{}")])])
    with pytest.raises(ToolCallError):
        NimChat(CFG, client).chat([{"role": "user", "content": "go"}], TOOLS)


def test_no_tools_omits_tools_param():
    client = FakeClient([reply(content="hello")])
    out = NimChat(CFG, client).chat([{"role": "user", "content": "hi"}], [])
    assert out.content == "hello" and out.tool_calls == ()
    assert "tools" not in client.requests[0]


def test_summarizer_uses_chat():
    client = FakeClient([reply(content="  Acme pays monthly. ")])
    assert NimSummarizer(NimChat(CFG, client)).summarize(["a", "b"]) == "Acme pays monthly."
