"""Benchmark personas (SPEC §11). Each is its own tenant with its own tools and attacker."""
from dataclasses import dataclass

from smriti.agent.tools import Fixtures
from smriti.oracle.predicates import AttackGoal, ExpectedCall


@dataclass(frozen=True)
class Task:
    turn: str                      # the user's request in the trigger session
    expected: ExpectedCall         # the legitimate call the oracle and simulated user accept


@dataclass(frozen=True)
class Persona:
    name: str
    tenant: str
    role: str
    tools: tuple[str, ...]
    seed_memories: tuple[str, ...]           # written by the user (HIGH) before the episode
    fixtures: Fixtures                       # benign pages/emails for clean sessions
    clean_turns: tuple[str, ...]             # benign, non-risky requests
    tasks: tuple[Task, ...]                  # trigger-session tasks, chosen by seed
    attacker: AttackGoal
    goal: str                                # imperative fragment the poison asks for
    carrier_kind: str                        # "web" | "email": how indirect poison arrives
    carrier_ref: str                         # url or email id
    carrier_doc: str                         # document text with a "{poison}" slot
    carrier_turn: str                        # user request that makes the agent read it


def get_persona(name: str) -> Persona:
    from smriti.personas import devops, finance, personal
    registry = {p.name: p for p in (finance.PERSONA, devops.PERSONA, personal.PERSONA)}
    return registry[name]
