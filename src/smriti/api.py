"""Public API: SMRITI as the memory of any agent harness.

    from smriti import Smriti

    mem = Smriti(tenant="acme")
    mem.remember(page_text, source="tool:web")      # write gate
    ctx = mem.recall("send the October invoice")    # read gate; starts a new turn
    system_prompt = BASE_PROMPT + ctx.prompt        # trusted memory as instructions, rest as data

    @mem.tool(risky=True)                           # action check (+ opening sealed PII)
    def send_email(to, subject, body): ...

    @mem.tool(source="tool:web")                    # its output is untrusted: taints the turn
    def read_web(url): ...

A thin layer over Gateway: no security logic lives here. Two hooks are needed for the
full guarantee: memory (remember/recall) and tools (the decorator). With only the memory
hook, untrusted memory is labeled and quoted, but risky actions are not stopped.

`protect=False` gives the same agent without SMRITI (the A0 arm), for before/after runs.
"""
import functools
import os
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

from smriti.agent.prompts import memory_sections
from smriti.agent.tools import _has_sealed, _unseal
from smriti.gateway.action_check import ActionDecision
from smriti.gateway.gateway import A0, A2, Gateway, GatewayConfig, WriteResult
from smriti.gateway.labels import ToolCall, Trust
from smriti.gateway.read_gate import RenderedContext

__all__ = ["Smriti", "Context", "DENIED", "UNSEAL_FAILED"]

# What a refused call returns instead of running; pass it back to the model as the result.
DENIED = "The user denied this action."
UNSEAL_FAILED = "Error: a private value could not be unsealed."


@dataclass(frozen=True)
class Context:
    """What recall() returns. Put `prompt` into the agent's system prompt."""
    prompt: str                    # ready-made memory section ("" when memory is empty)
    instructions: str              # HIGH-trust memory only
    data: str                      # everything else, escaped and wrapped as quoted data
    tainted: bool                  # untrusted memory is in context
    memory_ids: tuple[str, ...]


def _refuse(tool: str, args: dict) -> bool:
    return False


class Smriti:
    def __init__(
        self,
        tenant: str = "default",
        *,
        protect: bool = True,
        store=None,
        confirm: Callable[[str, dict], bool] = _refuse,
        embedder=None,
        judge=None,
        pii=None,
        summarizer=None,
        config: GatewayConfig | None = None,
        session: str | None = None,
        k: int = 8,
    ):
        """
        tenant:     memories are isolated per tenant.
        protect:    False = no SMRITI (plain memory, no checks), for comparisons.
        store:      None = in memory; a Postgres DSN or "postgresql://..." URL = persistent;
                    or any Store object.
        confirm:    asks the human about a risky action; True = approve. The default
                    refuses (fail closed).
        embedder, judge, pii, summarizer: model clients; None = sensible defaults (NVIDIA
                    hosted models when NVIDIA_API_KEY is set, local fallbacks otherwise).
        config:     a GatewayConfig to override every setting (e.g. risky_tools).
        """
        self.tenant = tenant
        self.session = session or uuid.uuid4().hex[:8]
        self.confirm = confirm
        self.k = k
        if config is None:
            config = A2 if protect else A0
        self.gateway = Gateway(_store(store), embedder or _embedder(), pii or _pii(),
                               judge or _judge(), summarizer, _with_judge(config, judge))
        self._turn = 0
        self._ctx = RenderedContext("", "", False, [])
        self._untrusted_output = False

    @property
    def config(self) -> GatewayConfig:
        return self.gateway.config

    # -- memory hook -----------------------------------------------------------------------

    def remember(self, text: str, source: str = "user") -> WriteResult:
        """Write through the write gate. `source` decides trust: "user" is HIGH,
        internal tools in config.med_sources are MED, everything else is LOW."""
        return self.gateway.write(text, source, self.tenant, self.session)

    def recall(self, query: str, k: int | None = None) -> Context:
        """Read through the read gate, and start a new turn: the action check then looks
        at this context plus any untrusted tool output seen until the next recall()."""
        ctx = self.gateway.read(query, self.tenant, k or self.k)
        self._turn += 1
        self._ctx = ctx
        self._untrusted_output = False
        return Context("\n\n".join(memory_sections(ctx)), ctx.instructions, ctx.data_block,
                       ctx.tainted, tuple(ctx.memory_ids))

    def consolidate(self, memory_ids: list[str]):
        """Summarize memories into one; the summary keeps the lowest trust of its inputs."""
        if self.gateway.summarizer is None:
            raise RuntimeError("pass summarizer=... to Smriti to consolidate")
        return self.gateway.consolidate(self.tenant, list(memory_ids), self.session)

    @property
    def tainted(self) -> bool:
        """Untrusted memory or untrusted tool output is in this turn's context."""
        return bool(self._ctx.tainted) or self._untrusted_output

    # -- tool hook -------------------------------------------------------------------------

    def tool(self, fn: Callable | None = None, *, risky: bool = False, source: str | None = None,
             name: str | None = None, remember_output: bool = False):
        """Decorate a tool the agent can call.

        risky=True: in a tainted turn the call needs confirm(); if refused it does not run
        and returns DENIED (pass that back to the model). A sealed value that cannot be
        opened returns UNSEAL_FAILED; the tool does not run. Also add the name to
        config.risky_tools to make it risky everywhere.
        source="tool:web": the tool's output comes from that source; if it is untrusted it
        taints the rest of the turn (remember_output=True also writes it to memory).
        Sealed PII ([PHONE_NUMBER:...]) in the arguments is opened only for the call itself.
        """
        def wrap(f: Callable) -> Callable:
            tool_name = name or f.__name__

            @functools.wraps(f)
            def guarded(*args, **kwargs):
                return self._call(tool_name, f, risky, source, remember_output, args, kwargs)
            guarded.smriti_tool = tool_name
            return guarded
        return wrap(fn) if fn is not None else wrap

    def _call(self, name, f, risky, source, remember_output, args, kwargs):
        call_args = dict(kwargs)
        if args:
            call_args["_args"] = list(args)
        ctx = replace(self._ctx, tainted=self.tainted)

        real_args, real_kwargs, types = args, kwargs, []
        if _has_sealed([args, kwargs]):
            def open_text(text: str):
                out = self.gateway.pii.unseal(text, self.tenant)
                return out.text, out.types
            try:
                real_args = tuple(_unseal(list(args), open_text, types))
                real_kwargs = _unseal(kwargs, open_text, types)
            except Exception:              # wrong tenant, altered value, no key: fail closed
                self._log(name, call_args, ctx, "rejected_fail_closed")
                return UNSEAL_FAILED

        needs_confirm = (self.gateway.check_action(name, call_args, ctx)
                         is ActionDecision.REQUIRE_CONFIRMATION)
        if risky and self.config.action_check and ctx.tainted:
            needs_confirm = True
        if types and ctx.tainted is not False:
            needs_confirm = True
        if needs_confirm:
            try:
                approved = bool(self.confirm(name, call_args))
            except Exception:
                approved = False                     # fail closed
            if not approved:
                self._log(name, call_args, ctx, "confirm_denied")
                return DENIED
        if types:
            try:
                self.gateway.store.append_ledger(self.tenant, "unseal", None, {
                    "tool": name, "types": types, "turn": self._turn})
            except Exception:              # an unseal that cannot be recorded does not happen
                self._log(name, call_args, ctx, "rejected_fail_closed")
                return UNSEAL_FAILED
        self._log(name, call_args, ctx, "executed")
        result = f(*real_args, **real_kwargs)

        if source is not None:
            if self.gateway.trust_of(source) == Trust.LOW:
                self._untrusted_output = True
            if remember_output and result is not None:
                self.remember(str(result), source)
        return result

    def _log(self, name: str, args: dict, ctx: RenderedContext, decision: str) -> None:
        try:
            self.gateway.store.log_tool_call(ToolCall(
                f"{self.tenant}/{self.session}", self.session, self._turn, name, args,
                bool(ctx.tainted), decision, list(ctx.memory_ids)))
        except Exception:
            pass                       # the decision is already made; logging is best effort


# -- defaults ------------------------------------------------------------------------------

_HOSTED = "https://integrate.api.nvidia.com/v1"
_JUDGE = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
_EMBED = "nvidia/nemotron-3.5-embed-1b"


def _store(store):
    if store is None:
        from smriti.store.fake_store import FakeStore
        return FakeStore()
    if isinstance(store, str):
        from smriti.store.store import PgStore
        pg = PgStore(store, schema="smriti", embed_dim=_embed_dim())
        pg.init_schema()
        return pg
    return store


def _embed_dim() -> int:
    return 2048 if os.environ.get("NVIDIA_API_KEY") else 256


def _embedder():
    if os.environ.get("NVIDIA_API_KEY"):
        from smriti.models.cache import DiskCache
        from smriti.models.hosted import CachedEmbedder, HostedConfig, NvidiaEmbedder
        return CachedEmbedder(NvidiaEmbedder(HostedConfig(_HOSTED, _EMBED)),
                              DiskCache(".cache/smriti"), _EMBED)
    warnings.warn("NVIDIA_API_KEY not set: using a simple local word-hash embedder "
                  "(fine for trying SMRITI, weak retrieval)", stacklevel=3)
    from smriti.models.hash_embed import HashEmbedder
    return HashEmbedder()


def _judge():
    if os.environ.get("NVIDIA_API_KEY"):
        from smriti.models.cache import DiskCache
        from smriti.models.hosted import CachedJudge, HostedConfig, SafetyGuardJudge
        return CachedJudge(SafetyGuardJudge(HostedConfig(_HOSTED, _JUDGE)),
                           DiskCache(".cache/smriti"), _JUDGE)
    return None


def _with_judge(config: GatewayConfig, judge) -> GatewayConfig:
    if judge is None and not os.environ.get("NVIDIA_API_KEY") and config.use_judge:
        warnings.warn("NVIDIA_API_KEY not set: the instruction detector uses heuristics only "
                      "(the guarantee does not depend on it)", stacklevel=3)
        return replace(config, use_judge=False)
    return config


def _pii():
    from smriti.models.pii import PiiSealer, PresidioRedactor
    sealer = PiiSealer.from_env()
    if sealer is None:
        warnings.warn("SMRITI_PII_KEY not set: phone numbers and names are removed for good "
                      "instead of encrypted", stacklevel=3)
    return PresidioRedactor(sealer=sealer)
