"""Counterfactual replay: re-run a past user turn, as of its original time, with some
memories hidden, and capture what the agent *tries* to do. Nothing is written: the store
is wrapped read-only, tool calls are captured instead of logged, and the agent runs dry.

Determinism comes from temperature 0. Known limits: the read gate's nonce differs between
runs, and flags are read as they are now (e.g. a memory archived since the turn stays
hidden); a replay whose baseline does not reproduce the original call is "inconclusive".
"""
import re
from dataclasses import dataclass

from smriti.agent.graph import Agent
from smriti.agent.tools import Fixtures, ToolRunner
from smriti.gateway.gateway import Gateway
from smriti.gateway.labels import ToolCall
from smriti.models.nim import ChatModel
from smriti.store.store import Store, Turn

# Fields that decide *where* a risky action goes; body/subject wording is ignored.
SIGNATURE_FIELDS = ("to", "cc", "bcc", "repo", "ref", "account", "payee")


@dataclass(frozen=True)
class Attempt:
    tool: str
    args: dict


@dataclass(frozen=True)
class AgentSpec:
    """What is needed to rebuild the agent for a replay."""
    role: str
    tools: tuple[str, ...]
    fixtures: Fixtures
    k: int = 8


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def destinations(args: dict) -> set[str]:
    """Where a call sends things: recipients, account, payee, repo, ref (normalized)."""
    out: set[str] = set()
    for f in SIGNATURE_FIELDS:
        values = args.get(f)
        for v in values if isinstance(values, (list, tuple)) else [values]:
            if v in (None, ""):
                continue
            text = str(v).strip().lower()
            m = _EMAIL.search(text)
            out.add(m.group(0) if m else text)
    return out


def signature(tool: str, args: dict) -> tuple:
    def norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(sorted(str(x).strip().lower() for x in v))
        return str(v).strip().lower()
    return (tool, tuple((f, norm(args[f])) for f in SIGNATURE_FIELDS if args.get(f)))


class _ReplayStore:
    """Read-through view of the real store: hides `exclude`, sees only memories created by
    `as_of`, captures tool calls, and ignores ledger writes."""

    def __init__(self, inner: Store, exclude: tuple[str, ...], as_of: float | None):
        self.inner, self.exclude, self.as_of = inner, exclude, as_of
        self.attempts: list[Attempt] = []

    def search(self, tenant, embedding, k, exclude_ids=(), as_of=None):
        return self.inner.search(tenant, embedding, k,
                                 exclude_ids=(*self.exclude, *exclude_ids),
                                 as_of=self.as_of if as_of is None else as_of)

    def get(self, tenant, memory_id):
        return self.inner.get(tenant, memory_id)

    def log_tool_call(self, call: ToolCall) -> None:
        self.attempts.append(Attempt(call.tool, dict(call.args)))

    def append_ledger(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, name):
        raise RuntimeError(f"replay store is read-only ({name})")


def replay_turn(turn: Turn, gateway: Gateway, llm: ChatModel, spec: AgentSpec,
                exclude: tuple[str, ...] = (), as_of: float | None = None) -> list[Attempt]:
    view = _ReplayStore(gateway.store, tuple(exclude), turn.ts if as_of is None else as_of)
    gw = Gateway(view, gateway.embedder, gateway.pii, gateway.judge, gateway.summarizer,
                 gateway.config)
    # Approve every confirmation: the question is what the agent would *try* to do.
    tools = ToolRunner(gw, spec.fixtures, lambda tool, args: True, spec.tools)
    agent = Agent(gw, llm, tools, spec.role, turn.tenant, k=spec.k, record=False)
    agent.run_turn(turn.user_text, turn.session_id, f"replay:{turn.seq}")
    return view.attempts
