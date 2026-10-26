"""Agent loop per user turn (SPEC §10): retrieve -> think -> act -> ... -> remember."""
import json
import re
import secrets
import time
from dataclasses import dataclass, replace
from typing import TypedDict

from langgraph.graph import END, StateGraph

from smriti.agent.prompts import system_prompt
from smriti.agent.tools import READ_SOURCES, ToolRunner
from smriti.gateway.gateway import Gateway
from smriti.gateway.labels import Trust
from smriti.gateway.read_gate import RenderedContext
from smriti.models.nim import ChatModel, LlmToolCall, ToolCallError

_REMEMBER_REQUEST = re.compile(r"^\s*remember\b", re.IGNORECASE)


class TurnState(TypedDict, total=False):
    user_text: str
    session_id: str
    episode_id: str
    turn_seq: int | None
    nonce: str
    ctx: RenderedContext
    messages: list[dict]
    pending: tuple[LlmToolCall, ...]
    step: int
    low_read: bool                 # a LOW-trust tool output entered this turn's context
    reads: list[tuple[str, str]]   # (source, text) tool outputs to remember
    final: str


@dataclass(frozen=True)
class TurnResult:
    reply: str
    steps: int


class Agent:
    def __init__(self, gateway: Gateway, llm: ChatModel, tools: ToolRunner, role: str,
                 tenant: str, k: int = 8, max_steps: int = 6, record: bool = True):
        self.gateway = gateway
        self.llm = llm
        self.tools = tools
        self.role = role
        self.tenant = tenant
        self.k = k
        self.max_steps = max_steps
        # record=False is the auditor's dry replay: no turn log, nothing remembered.
        self.record = record
        self._graph = self._build()

    def run_turn(self, user_text: str, session_id: str, episode_id: str,
                 nonce: str | None = None) -> TurnResult:
        """`nonce` is only passed by the auditor's replay, to reproduce the original prompt."""
        nonce = nonce or secrets.token_hex(8)
        turn_seq = (self._record_turn(user_text, session_id, episode_id, nonce)
                    if self.record else None)
        out = self._graph.invoke(
            {"user_text": user_text, "session_id": session_id, "episode_id": episode_id,
             "turn_seq": turn_seq, "nonce": nonce, "step": 0, "low_read": False, "reads": [],
             "final": ""},
            config={"recursion_limit": 2 * self.max_steps + 10})
        return TurnResult(out["final"], out["step"])

    def _record_turn(self, user_text: str, session_id: str, episode_id: str,
                     nonce: str) -> int | None:
        """Log the request (PII-redacted or sealed) so the Immune Auditor can replay it later.
        Best effort: a logging failure must never block the user's turn."""
        try:
            text = self.gateway.pii.redact(user_text, self.tenant).text
            return self.gateway.store.record_turn(self.tenant, episode_id, session_id, text,
                                                  time.time(), nonce)
        except Exception:
            return None

    def _build(self):
        g = StateGraph(TurnState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("think", self._think)
        g.add_node("act", self._act)
        g.add_node("remember", self._remember)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "think")
        g.add_conditional_edges("think", lambda s: "act" if s["pending"] else "remember")
        g.add_edge("act", "think")
        g.add_edge("remember", END)
        return g.compile()

    def _retrieve(self, s: TurnState) -> dict:
        ctx = self.gateway.read(s["user_text"], self.tenant, self.k, nonce=s["nonce"])
        return {"ctx": ctx, "messages": [
            {"role": "system", "content": system_prompt(
                self.role, ctx, guard=self.gateway.config.prompt_guard)},
            {"role": "user", "content": s["user_text"]}]}

    def _think(self, s: TurnState) -> dict:
        if s["step"] >= self.max_steps:
            return {"pending": (), "final": "Stopped: step limit reached."}
        try:
            reply = self.llm.chat(s["messages"], self.tools.schemas())
        except ToolCallError:
            return {"pending": (), "final": "Stopped: invalid tool call."}
        msg: dict = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                for c in reply.tool_calls]
        return {"messages": s["messages"] + [msg], "pending": reply.tool_calls,
                "final": reply.content or ""}

    def _act(self, s: TurnState) -> dict:
        messages, reads, low_read = list(s["messages"]), list(s["reads"]), s["low_read"]
        for call in s["pending"]:
            ctx = replace(s["ctx"], tainted=s["ctx"].tainted or low_read)
            out = self.tools.run(call.name, call.args, ctx, episode_id=s["episode_id"],
                                 session_id=s["session_id"], step=s["step"],
                                 turn_seq=s.get("turn_seq"), tenant=self.tenant)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out.text})
            source = READ_SOURCES.get(call.name)
            if source and out.decision == "executed":
                reads.append((source, out.text))
                low_read = low_read or self.gateway.trust_of(source) == Trust.LOW
        return {"messages": messages, "reads": reads, "low_read": low_read,
                "pending": (), "step": s["step"] + 1}

    def _remember(self, s: TurnState) -> dict:
        if not self.record:
            return {}
        for source, text in s["reads"]:
            self.gateway.write(text, source, self.tenant, s["session_id"])
        if _REMEMBER_REQUEST.match(s["user_text"]):
            self.gateway.write(s["user_text"], "user", self.tenant, s["session_id"])
        return {}
