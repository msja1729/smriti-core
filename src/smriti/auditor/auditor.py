"""Immune Auditor: find memories that steered risky actions, roll them back, and list the
actions they touched.

For every risky tool call (executed or denied) the suspects are the memories in its
context that the user did not write (judged by source, so this works even without trust
labels). Each suspect is hidden in a counterfactual replay of the turn; if the risky call
disappears or changes destination, that memory was steering the agent. If no single
suspect explains the call (e.g. the poison exists twice: original + summary), all suspects
are hidden together and the set is shrunk to a minimal one that still explains it. The
decision is made by comparing call signatures in code, never by an LLM.
"""
from dataclasses import dataclass, field

from smriti.auditor.replay import AgentSpec, Attempt, replay_turn, signature
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
            sig = signature(call.tool, call.args)
            baseline = self._replay(turn, (), report)
            if sig not in {signature(a.tool, a.args) for a in baseline}:
                report.findings.append(Finding(turn.seq, call.tool, call.args, None,
                                               "inconclusive"))
                continue
            suspects = [mid for mid in call.memory_ids_in_context
                        if (m := store.get(tenant, mid)) is not None
                        and m.label.source != "user"]
            steering = [mid for mid in suspects if not self._reproduces(turn, (mid,), sig, report)]
            if not steering and suspects and not self._reproduces(turn, tuple(suspects), sig,
                                                                  report):
                steering = self._minimal_set(turn, suspects, sig, report)
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

    def _reproduces(self, turn, exclude: tuple[str, ...], sig: tuple,
                    report: AuditReport) -> bool:
        return sig in {signature(a.tool, a.args) for a in self._replay(turn, exclude, report)}

    def _minimal_set(self, turn, suspects: list[str], sig: tuple,
                     report: AuditReport) -> list[str]:
        """Hiding all suspects removes the call; put each back in turn and keep it hidden only
        if putting it back brings the call back."""
        hidden = list(suspects)
        for mid in suspects:
            trial = tuple(x for x in hidden if x != mid)
            if not self._reproduces(turn, trial, sig, report):
                hidden = list(trial)        # not needed to explain the call
        return hidden

    def _replay(self, turn, exclude: tuple[str, ...], report: AuditReport) -> list[Attempt]:
        key = (turn.seq, exclude)
        if key not in self._cache:
            self._cache[key] = replay_turn(turn, self.gateway, self.llm, self.spec, exclude)
            report.replays += 1
        return self._cache[key]
