"""SMRITI gateway (SPEC §8): the only path between the agent and memory. Fail-closed:
write errors -> rejected; read errors -> HIGH-only; action-check errors -> confirmation."""
import secrets
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace

from smriti.gateway.action_check import ActionDecision, check_action
from smriti.gateway.consolidate import Summarizer, summary_label
from smriti.gateway.heuristics import HeuristicConfig
from smriti.gateway.labels import Label, Memory, Sensitivity, Trust
from smriti.gateway.read_gate import RenderedContext, render
from smriti.gateway.write_gate import WriteResult, screen, trust_for
from smriti.models.hosted import Embedder, Judge
from smriti.models.pii import PiiRedactor
from smriti.store.store import Store

__all__ = ["Gateway", "GatewayConfig", "WriteResult", "RenderedContext", "ActionDecision"]


@dataclass(frozen=True)
class GatewayConfig:
    # Defense switches for arms/ablations (SPEC §8); all on = A2.
    label_provenance: bool = True
    write_flagging: bool = True
    read_escaping: bool = True
    action_check: bool = True
    use_judge: bool = True
    # A1 baseline only: judge every input (no heuristics, no provenance) and add a
    # prompt line telling the model to ignore instructions found in memory.
    judge_all_inputs: bool = False
    prompt_guard: bool = False
    archive_consolidated: bool = True
    quota_low_writes_per_session: int = 20
    med_sources: frozenset[str] = frozenset({"tool:calendar", "tool:crm"})
    risky_tools: frozenset[str] = frozenset({"send_email", "pay", "deploy", "share_file"})
    heuristics: HeuristicConfig = field(default_factory=HeuristicConfig)


# Arm presets (SPEC §10). A0: plain store, everything in one prompt section, no defense.
A0 = GatewayConfig(label_provenance=False, write_flagging=False, read_escaping=False,
                   action_check=False, use_judge=False, quota_low_writes_per_session=10**9)
A2 = GatewayConfig()


class Gateway:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        pii: PiiRedactor,
        judge: Judge,
        summarizer: Summarizer,
        config: GatewayConfig = GatewayConfig(),
    ):
        self.store = store
        self.embedder = embedder
        self.pii = pii
        self.judge = judge
        self.summarizer = summarizer
        self.config = config
        self._low_writes: Counter[tuple[str, str, str]] = Counter()

    # -- write -------------------------------------------------------------------------

    def write(self, text: str, source: str, tenant: str, session_id: str) -> WriteResult:
        start = time.perf_counter()
        try:
            return self._write(text, source, tenant, session_id, start)
        except Exception as e:
            self._ledger_best_effort(tenant, "reject", None,
                                     {"reason": "fail_closed", "error": type(e).__name__})
            return WriteResult("rejected", None, frozenset({"fail_closed"}), _ms(start))

    def _write(self, text, source, tenant, session_id, start) -> WriteResult:
        cfg = self.config
        trust = self.trust_of(source)
        label = Label(source, tenant, trust, Sensitivity.PUBLIC, frozenset())
        s = self._screen(text, label)

        quota_key = (tenant, session_id, source)
        if trust == Trust.LOW and self._low_writes[quota_key] >= cfg.quota_low_writes_per_session:
            self.store.append_ledger(tenant, "reject", None, {**s.detail, "reason": "quota_hit"})
            return WriteResult("rejected", None, s.flags | {"quota_hit"}, _ms(start))

        memory = Memory(str(uuid.uuid4()), s.text, s.label, s.flags, session_id, time.time())
        self._store(memory)
        if trust == Trust.LOW:
            self._low_writes[quota_key] += 1
        status = "quarantined" if "quarantined" in s.flags else "stored"
        self.store.append_ledger(tenant, "quarantine" if status == "quarantined" else "write",
                                 memory.id, {**s.detail, "flags": sorted(s.flags),
                                             "latency_ms": _ms(start)})
        return WriteResult(status, memory.id, s.flags, _ms(start))

    def trust_of(self, source: str) -> Trust:
        """Trust a source gets under this config (everything HIGH without provenance)."""
        if not self.config.label_provenance:
            return Trust.HIGH
        return trust_for(source, self.config.med_sources)

    def _screen(self, text: str, label: Label):
        cfg = self.config
        return screen(text, label, flagging=cfg.write_flagging, use_judge=cfg.use_judge,
                      judge_all=cfg.judge_all_inputs, heuristics=cfg.heuristics,
                      judge=self.judge, pii=self.pii)

    def _store(self, memory: Memory) -> None:
        # Quarantined memories are never retrieved, so they need no embedding.
        embedding = None if "quarantined" in memory.flags else self.embedder.embed([memory.text], "passage")[0]
        self.store.insert(memory, embedding)

    # -- consolidate -------------------------------------------------------------------

    def consolidate(self, tenant: str, memory_ids: list[str], session_id: str) -> Memory:
        try:
            inputs = [self.store.get(tenant, mid) for mid in memory_ids]
            if any(m is None for m in inputs):
                raise KeyError("memory not found in tenant")
            label = summary_label(tenant, inputs)
            text = self.summarizer.summarize([m.text for m in inputs])
            s = self._screen(text, label)
            # Screening may raise sensitivity (new PII) but never trust: label stays the join.
            memory = Memory(str(uuid.uuid4()), s.text, s.label, s.flags, session_id, time.time())
            self._store(memory)
            if self.config.archive_consolidated:
                self.store.archive(tenant, list(memory_ids))
        except Exception as e:
            self._ledger_best_effort(tenant, "reject", None,
                                     {"reason": "consolidate_failed", "error": type(e).__name__})
            raise
        self.store.append_ledger(tenant, "summarize", memory.id, {
            **s.detail, "inputs": len(inputs), "flags": sorted(s.flags)})
        return memory

    # -- read --------------------------------------------------------------------------

    def read(self, query: str, tenant: str, k: int = 8) -> RenderedContext:
        start = time.perf_counter()
        results: list[Memory] = []
        try:
            results = self.store.search(tenant, self.embedder.embed([query], "query")[0], k)
            if any(m.label.tenant != tenant for m in results):
                raise RuntimeError("store returned another tenant's memory")
            ctx = render(results, secrets.token_hex(8), escaping=self.config.read_escaping)
            self.store.append_ledger(tenant, "read", None, {
                "memory_ids": ctx.memory_ids, "latency_ms": _ms(start)})
            return ctx
        except Exception as e:
            high = [m for m in results if m.label.tenant == tenant and m.label.trust == Trust.HIGH]
            ctx = RenderedContext("\n".join(f"- {m.text}" for m in high), "", False,
                                  [m.id for m in high])
            self._ledger_best_effort(tenant, "read", None, {
                "memory_ids": ctx.memory_ids, "reason": "fail_closed",
                "error": type(e).__name__})
            return ctx

    # -- action check ------------------------------------------------------------------

    def check_action(self, tool: str, args: dict, ctx: RenderedContext) -> ActionDecision:
        try:
            if not self.config.action_check:
                return ActionDecision.ALLOW
            if not isinstance(ctx.tainted, bool):
                raise TypeError("tainted must be a bool")
            return check_action(tool, ctx.tainted, self.config.risky_tools)
        except Exception:
            return ActionDecision.REQUIRE_CONFIRMATION

    # -- helpers -----------------------------------------------------------------------

    def _ledger_best_effort(self, tenant: str, op: str, memory_id: str | None, detail: dict):
        try:
            self.store.append_ledger(tenant, op, memory_id, detail)
        except Exception:
            pass                   # the decision is already fail-closed; never mask it


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
