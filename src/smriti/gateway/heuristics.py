"""Deterministic instruction-likeness scoring (SPEC §9.1 step 2). No model calls."""
import re
from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    CLEAN = "clean"
    UNCERTAIN = "uncertain"            # goes to the judge
    INSTRUCTION_LIKE = "instruction_like"


@dataclass(frozen=True)
class HeuristicConfig:
    clean_below: float = 1.0           # score < clean_below -> CLEAN
    flag_at: float = 2.5               # score >= flag_at -> INSTRUCTION_LIKE

    def __post_init__(self) -> None:
        if not 0 < self.clean_below <= self.flag_at:
            raise ValueError("need 0 < clean_below <= flag_at")


@dataclass(frozen=True)
class HeuristicResult:
    score: float
    verdict: Verdict
    matches: tuple[str, ...]


_I = re.IGNORECASE

# (name, weight, pattern). A rule counts once however often it matches.
_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    ("future_imperative", 1.0, re.compile(
        r"\b(always|from now on|going forward|in (the )?future|make sure to|be sure to)\b", _I)),
    ("never", 0.5, re.compile(r"\bnever\b", _I)),
    ("remember", 1.0, re.compile(r"\bremember\s*(to\b|:|that\b)", _I)),
    ("meta_instruction", 2.5, re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above|earlier)\b"
        r"|\byou\s+must\b|\bas an ai\b|^\s*system\s*:|\bnew instructions?\b", _I | re.M)),
    ("trigger", 2.5, re.compile(
        r"\b(when|whenever|if|once)\s+(the\s+)?user\s+(says|asks|mentions|types)\b", _I)),
]

_ACTION = re.compile(
    r"\b(send|sent|cc|bcc|forward|pay|payments?|transfer|wire|deploy|share)\b", _I)
_TARGET = re.compile(
    r"[\w.+-]+@[\w-]+(\.[\w-]+)+"                 # email address
    r"|\biban\b|\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"  # IBAN
    r"|\b[\w.-]+/[\w.-]+\b(?=.*\b(repo|branch)\b)|\bfrom\s+[\w.-]+/[\w.-]+",  # repo
    _I)
_ACTION_TARGET_WEIGHT = 1.5


def score(text: str, config: HeuristicConfig = HeuristicConfig()) -> HeuristicResult:
    matches: list[str] = []
    total = 0.0
    for name, weight, pattern in _RULES:
        if pattern.search(text):
            matches.append(name)
            total += weight
    if _ACTION.search(text) and _TARGET.search(text):
        matches.append("action_target")
        total += _ACTION_TARGET_WEIGHT

    if total >= config.flag_at:
        verdict = Verdict.INSTRUCTION_LIKE
    elif total < config.clean_below:
        verdict = Verdict.CLEAN
    else:
        verdict = Verdict.UNCERTAIN
    return HeuristicResult(total, verdict, tuple(matches))
