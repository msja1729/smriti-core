"""Consolidation labels (SPEC §9.2). The label is computed here, never by the LLM.
Mirrors lean/Smriti/Label.lean `summarize`; proved there: `summary_inherits` and
`summary_low_of_low_input` (any LOW input makes the summary LOW)."""
from functools import reduce
from typing import Protocol

from smriti.gateway.labels import Label, Memory, join, top


class Summarizer(Protocol):
    def summarize(self, texts: list[str]) -> str: ...


def summary_label(tenant: str, inputs: list[Memory]) -> Label:
    label = reduce(join, (m.label for m in inputs), top(tenant))
    # Invariant (§6): parents = every input's parents plus every input's own id.
    return Label(label.source, label.tenant, label.trust, label.sensitivity,
                 label.parents | {m.id for m in inputs})
