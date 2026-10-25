"""Provenance labels (SPEC §6). Mirrors lean/Smriti/Label.lean: `Label.join`, `Trust`."""
from dataclasses import dataclass
from enum import IntEnum


class Trust(IntEnum):
    LOW = 0
    MED = 1
    HIGH = 2


class Sensitivity(IntEnum):        # IntEnum, never strings:
    PUBLIC = 0                     # max("public", "pii") == "public" would launder PII
    PII = 1


@dataclass(frozen=True)
class Label:
    source: str                    # "user" | "tool:web" | "tool:email" | "tool:<name>" | "agent" | "derived"
    tenant: str
    trust: Trust
    sensitivity: Sensitivity
    parents: frozenset[str]        # ids of memories this was derived from


class CrossTenantError(Exception): ...


def join(a: Label, b: Label) -> Label:
    """Lean: `Label.join` -- min trust, max sensitivity, union of parents."""
    if a.tenant != b.tenant:
        raise CrossTenantError(a.tenant, b.tenant)
    return Label(
        source="derived",
        tenant=a.tenant,
        trust=min(a.trust, b.trust),
        sensitivity=max(a.sensitivity, b.sensitivity),
        parents=a.parents | b.parents,
    )


def top(tenant: str) -> Label:     # identity element for join
    return Label("derived", tenant, Trust.HIGH, Sensitivity.PUBLIC, frozenset())


@dataclass(frozen=True)
class Memory:
    id: str                        # uuid
    text: str                      # PII already redacted
    label: Label
    flags: frozenset[str]          # {"instruction_like", "quarantined", "quota_hit", ...}
    session_id: str
    created_at: float


@dataclass(frozen=True)
class ToolCall:                    # written ONLY by our tool wrappers; the oracle reads only these
    episode_id: str
    session_id: str
    step: int
    tool: str
    args: dict
    tainted: bool                  # any LOW memory in this step's context
    decision: str                  # "executed" | "confirm_denied" | "rejected_fail_closed"
    memory_ids_in_context: list[str]   # for blast-radius tracing
    turn_seq: int | None = None        # the user turn that produced the call (auditor replay)
