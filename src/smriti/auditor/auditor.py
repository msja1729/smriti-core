"""Immune Auditor: find memories that steered risky actions, roll them back, and list the
actions they touched.

Low-trust memory may *inform* an action the user asked for, so only **unsanctioned
destinations** are chased: recipients, accounts, repos or refs of a risky call (executed or
denied) that appear nowhere in what the user wrote (their request, or user-written memories
in context). The suspects are the memories in that call's context the user did not write
(judged by source, so this works even without trust labels) and whose text mentions the
unsanctioned destination: that is where it can have come from. Only if none mentions it
(the model composed it) are all non-user memories suspects. Each suspect is hidden in a
counterfactual replay of the turn; if the unsanctioned destination disappears, that memory
was steering the agent. If no single suspect explains it (e.g. the poison exists twice:
original + summary), all suspects are hidden together and the set is shrunk to a minimal
one. Every decision is made by comparing destinations in code, never by an LLM.
"""
from dataclasses import dataclass, field

from smriti.auditor.replay import AgentSpec, Attempt, destinations, replay_turn
from smriti.gateway.gateway import Gateway
from smriti.gateway.labels import ToolCall
from smriti.models.nim import ChatModel

DECISIONS_TO_AUDIT = ("executed", "confirm_denied")


@dataclass(frozen=True)
class Finding:
    turn_seq: int
    tool: str
    args: dict
    suspect: str | None            # None when the baseline replay was inconclusive
    verdict: str                   # "steering" | "not_steering" | "inconclusive"


@dataclass
class AuditReport:
    tenant: str
    episode_id: str
    findings: list[Finding] = field(default_factory=list)
    steering: list[str] = field(default_factory=list)       # memories that steered a call
    rolled_back: list[str] = field(default_factory=list)    # steering + their descendants
    review: list[ToolCall] = field(default_factory=list)    # executed calls they touched
    replays: int = 0


class ImmuneAuditor:
    def __init__(self, gateway: Gateway, llm: ChatModel, spec: AgentSpec):
        self.gateway = gateway
        self.llm = llm
        self.spec = spec
        self._cache: dict[tuple[int, tuple[str, ...]], list[Attempt]] = {}

    def audit(self, tenant: str, episode_id: str, rollback: bool = True) -> AuditReport:
        store = self.gateway.store
        report = AuditReport(tenant, episode_id)
        turns = {t.seq: t for t in store.turns(tenant, episode_id)}
        risky = self.gateway.config.risky_tools
        log = store.tool_log(episode_id)
        calls = [c for c in log if c.tool in risky and c.decision in DECISIONS_TO_AUDIT
                 and c.turn_seq in turns]

        for call in calls:
            turn = turns[call.turn_seq]
            context = [m for mid in call.memory_ids_in_context
                       if (m := store.get(tenant, mid)) is not None]
            user_text = " ".join([turn.user_text, *(m.text for m in context
                                                    if m.label.source == "user")]).lower()
            bad = frozenset(d for d in destinations(call.args) if d not in user_text)
            if not bad:
                continue                       # user-sanctioned: memory only informed it
            if not self._reproduces(turn, (), bad, report):
                report.findings.append(Finding(turn.seq, call.tool, call.args, None,
                                               "inconclusive"))
                continue
            others = [m for m in context if m.label.source != "user"]
            mentions = [m.id for m in others if any(d in m.text.lower() for d in bad)]
            suspects = mentions or [m.id for m in others]
            steering = [mid for mid in suspects if not self._reproduces(turn, (mid,), bad, report)]
            if not steering and suspects and not self._reproduces(turn, tuple(suspects), bad,
                                                                  report):
                steering = self._minimal_set(turn, suspects, bad, report)
            for mid in suspects:
                report.findings.append(Finding(turn.seq, call.tool, call.args, mid,
                                               "steering" if mid in steering else "not_steering"))
            for mid in steering:
                if mid not in report.steering:
                    report.steering.append(mid)

        if report.steering:
            descendants = [m.id for m in store.descendants(tenant, report.steering)]
            report.rolled_back = list(dict.fromkeys(report.steering + descendants))
            if rollback:
                store.quarantine(tenant, report.rolled_back)
                for mid in report.rolled_back:
                    reason = ("counterfactual_steering" if mid in report.steering
                              else "descendant_of_steering_memory")
                    store.append_ledger(tenant, "rollback", mid, {
                        "reason": reason, "episode_id": episode_id,
                        "evidence": [{"turn_seq": f.turn_seq, "tool": f.tool}
                                     for f in report.findings
                                     if f.suspect == mid and f.verdict == "steering"]})
        bad = set(report.rolled_back)
        report.review = [c for c in log if c.decision == "executed" and c.tool in risky
                         and bad & set(c.memory_ids_in_context)]
        return report

    def _reproduces(self, turn, exclude: tuple[str, ...], bad: frozenset[str],
                    report: AuditReport) -> bool:
        """Does any attempted risky call still go to one of the unsanctioned destinations?"""
        risky = self.gateway.config.risky_tools
        return any(bad & destinations(a.args) for a in self._replay(turn, exclude, report)
                   if a.tool in risky)

    def _minimal_set(self, turn, suspects: list[str], bad: frozenset[str],
                     report: AuditReport) -> list[str]:
        """Hiding all suspects removes the call; put each back in turn and keep it hidden only
        if putting it back brings the call back."""
        hidden = list(suspects)
        for mid in suspects:
            trial = tuple(x for x in hidden if x != mid)
            if not self._reproduces(turn, trial, bad, report):
                hidden = list(trial)        # not needed to explain the call
        return hidden

    def _replay(self, turn, exclude: tuple[str, ...], report: AuditReport) -> list[Attempt]:
        key = (turn.seq, exclude)
        if key not in self._cache:
            self._cache[key] = replay_turn(turn, self.gateway, self.llm, self.spec, exclude)
            report.replays += 1
        return self._cache[key]
