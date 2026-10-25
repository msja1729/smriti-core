"""Write gate (SPEC §9.1): label, flag, judge, quarantine, redact. Quota + ledger live in
gateway.py because they need session state and the store."""
from dataclasses import dataclass, replace

from smriti.gateway.heuristics import HeuristicConfig, Verdict, score
from smriti.gateway.labels import Label, Sensitivity, Trust
from smriti.models.hosted import Judge
from smriti.models.pii import PiiRedactor


@dataclass
class WriteResult:
    status: str                    # "stored" | "quarantined" | "rejected"
    memory_id: str | None
    flags: frozenset[str]
    latency_ms: float


@dataclass(frozen=True)
class Screened:
    text: str                      # PII-redacted
    label: Label
    flags: frozenset[str]
    detail: dict                   # for the ledger; never contains raw text


def trust_for(source: str, med_sources: frozenset[str]) -> Trust:
    """Default trust by source (SPEC §6). Unknown sources are LOW."""
    if source == "user":
        return Trust.HIGH
    if source in med_sources:
        return Trust.MED
    return Trust.LOW


def screen(
    text: str,
    label: Label,
    *,
    flagging: bool,
    use_judge: bool,
    judge_all: bool = False,
    heuristics: HeuristicConfig,
    judge: Judge,
    pii: PiiRedactor,
) -> Screened:
    flags: set[str] = set()
    detail: dict = {"source": label.source, "trust": label.trust.name}

    if judge_all:
        # A1 baseline: every input goes to the judge; unsafe is dropped whatever its source.
        unsafe = judge.is_unsafe(text)
        detail["judge_unsafe"] = unsafe
        if unsafe:
            flags |= {"instruction_like", "quarantined"}
    elif flagging:
        h = score(text, heuristics)
        detail.update(heuristic_score=h.score, heuristic_matches=list(h.matches))
        verdict = h.verdict
        if verdict is Verdict.UNCERTAIN and use_judge:
            unsafe = judge.is_unsafe(text)
            detail["judge_unsafe"] = unsafe
            verdict = Verdict.INSTRUCTION_LIKE if unsafe else Verdict.CLEAN
        if verdict is Verdict.INSTRUCTION_LIKE:
            flags.add("instruction_like")
            if label.trust < Trust.HIGH:
                flags.add("quarantined")

    redacted = pii.redact(text)
    detail["pii"] = sorted(redacted.found)
    if redacted.found:
        label = replace(label, sensitivity=Sensitivity.PII)

    return Screened(redacted.text, label, frozenset(flags), detail)
