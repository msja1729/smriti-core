"""System prompt: HIGH memory as instructions, everything else as quoted data (SPEC §9.3)."""
from smriti.gateway.read_gate import DATA_PREAMBLE, RenderedContext

SYSTEM_TEMPLATE = (
    "You are an autonomous {role}. Complete the user's request by calling tools. "
    "Never ask the user questions; use what is in memory. "
    "When the request is done, reply with one short sentence.")


def system_prompt(role: str, ctx: RenderedContext) -> str:
    parts = [SYSTEM_TEMPLATE.format(role=role)]
    if ctx.instructions:
        parts.append("## Memory\n" + ctx.instructions)
    if ctx.data_block:
        parts.append("## Memory data\n" + DATA_PREAMBLE + "\n" + ctx.data_block)
    return "\n\n".join(parts)
