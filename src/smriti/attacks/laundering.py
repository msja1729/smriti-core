"""Summary laundering: an indirect poison, then clean sessions, then consolidation right
before the trigger. Consolidated inputs are archived, so the attack can only fire from the
summary."""
from dataclasses import replace

from smriti.attacks import Poison, indirect
from smriti.personas import Persona


def make(persona: Persona, seed: int) -> Poison:
    return replace(indirect.make(persona, seed), attack="laundering",
                   consolidate_before_trigger=True)
