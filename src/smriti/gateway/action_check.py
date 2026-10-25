"""Action check (SPEC §9.4): a risky tool in a tainted step needs confirmation."""
from enum import Enum


class ActionDecision(Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"


def check_action(tool: str, tainted: bool, risky_tools: frozenset[str]) -> ActionDecision:
    if tainted and tool in risky_tools:
        return ActionDecision.REQUIRE_CONFIRMATION
    return ActionDecision.ALLOW
