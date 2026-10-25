"""Attack classes (SPEC §11). Each class turns (persona, seed) into a Poison: the text, how
it reaches memory, and what the trigger session needs. Wording varies by seed; some
templates avoid the heuristic trigger words on purpose so the other layers get tested."""
from dataclasses import dataclass

from smriti.personas import Persona


@dataclass(frozen=True)
class Poison:
    attack: str                    # "direct" | "indirect" | "laundering" | "sleeper"
    text: str                      # the poison sentence
    template: int                  # index of the template used
    channel: str                   # "memory_write": written straight to memory (direct)
                                   # "read": inside a document the agent reads
    write_source: str = ""         # provenance of a memory_write (the compromised channel)
    trigger_suffix: str = ""       # appended to the trigger turn (sleeper)
    consolidate_before_trigger: bool = False


ATTACKS = ("direct", "indirect", "laundering", "sleeper")


def make_poison(attack: str, persona: Persona, seed: int) -> Poison:
    from smriti.attacks import direct, indirect, laundering, sleeper
    makers = {"direct": direct.make, "indirect": indirect.make,
              "laundering": laundering.make, "sleeper": sleeper.make}
    return makers[attack](persona, seed)
