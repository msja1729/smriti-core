"""In-memory store with the same interface as PgStore, for unit tests (no DB)."""
import math
import time
from dataclasses import replace

from smriti.gateway.labels import CrossTenantError, Memory, ToolCall
from smriti.store.store import LedgerEntry, Turn


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else math.inf


class FakeStore:
    def __init__(self) -> None:
        self.reset()

    def insert(self, memory: Memory, embedding: list[float] | None) -> None:
        if memory.id in self._memories:
            raise ValueError(f"duplicate memory id {memory.id}")
        tenant = memory.label.tenant
        for parent in memory.label.parents:
            stored = self._memories.get(parent)
            if stored is None or stored[0].label.tenant != tenant:
                raise CrossTenantError(tenant, "parent outside tenant")
        self._memories[memory.id] = (memory, None if embedding is None else list(embedding))

    def get(self, tenant: str, memory_id: str) -> Memory | None:
        stored = self._memories.get(memory_id)
        if stored is None or stored[0].label.tenant != tenant:
            return None
        return stored[0]

    def search(self, tenant: str, embedding: list[float], k: int,
               exclude_ids: tuple[str, ...] = (), as_of: float | None = None) -> list[Memory]:
        candidates = [
            (_cosine_distance(emb, embedding), m)
            for m, emb in self._memories.values()
            if m.label.tenant == tenant and emb is not None
            and not m.flags & {"quarantined", "archived"}
            and m.id not in exclude_ids and (as_of is None or m.created_at <= as_of)
        ]
        candidates.sort(key=lambda c: c[0])
        return [m for _, m in candidates[:k]]

    def archive(self, tenant: str, memory_ids: list[str]) -> None:
        self._add_flag(tenant, memory_ids, "archived")

    def quarantine(self, tenant: str, memory_ids: list[str]) -> None:
        self._add_flag(tenant, memory_ids, "quarantined")

    def _add_flag(self, tenant: str, memory_ids: list[str], flag: str) -> None:
        for mid in memory_ids:
            stored = self._memories.get(mid)
            if stored is not None and stored[0].label.tenant == tenant:
                m, emb = stored
                self._memories[mid] = (replace(m, flags=m.flags | {flag}), emb)

    def descendants(self, tenant: str, memory_ids: list[str]) -> list[Memory]:
        ids = set(memory_ids)
        return [m for m, _ in self._memories.values()
                if m.label.tenant == tenant and m.label.parents & ids]

    def record_turn(self, tenant: str, episode_id: str, session_id: str, user_text: str,
                    ts: float) -> int:
        seq = len(self._turns) + 1
        self._turns.append(Turn(seq, ts, tenant, episode_id, session_id, user_text))
        return seq

    def turns(self, tenant: str, episode_id: str) -> list[Turn]:
        return [t for t in self._turns if t.tenant == tenant and t.episode_id == episode_id]

    def append_ledger(self, tenant: str, op: str, memory_id: str | None, detail: dict) -> None:
        self._ledger.append(
            LedgerEntry(len(self._ledger) + 1, time.time(), tenant, op, memory_id, dict(detail)))

    def ledger(self, tenant: str) -> list[LedgerEntry]:
        return [e for e in self._ledger if e.tenant == tenant]

    def log_tool_call(self, call: ToolCall) -> None:
        self._tool_log.append(call)

    def tool_log(self, episode_id: str) -> list[ToolCall]:
        return [c for c in self._tool_log if c.episode_id == episode_id]

    def reset(self) -> None:
        self._memories: dict[str, tuple[Memory, list[float] | None]] = {}
        self._ledger: list[LedgerEntry] = []
        self._tool_log: list[ToolCall] = []
        self._turns: list[Turn] = []
