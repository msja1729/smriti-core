"""Arms and ablations (SPEC §10) as gateway configs."""
from dataclasses import replace

from smriti.gateway.gateway import A0, A2, GatewayConfig

# A1: plain store + "ignore instructions in memory" + the judge on every input.
A1 = replace(A0, use_judge=True, judge_all_inputs=True, prompt_guard=True)

# A3 ablations: A2 minus one layer. no_write_flagging is the "detector off" run (§1).
ABLATIONS: dict[str, GatewayConfig] = {
    "no_write_flagging": replace(A2, write_flagging=False, use_judge=False),
    "no_read_escaping": replace(A2, read_escaping=False),
    "no_action_check": replace(A2, action_check=False),
    "no_provenance": replace(A2, label_provenance=False),
}

ARMS: dict[str, GatewayConfig] = {"A0": A0, "A1": A1, "A2": A2, **ABLATIONS}


def arm_config(name: str, quota_low_writes_per_session: int) -> GatewayConfig:
    config = ARMS[name]
    if name in ("A0", "A1"):       # plain store: no quota
        return config
    return replace(config, quota_low_writes_per_session=quota_low_writes_per_session)
