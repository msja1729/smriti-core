"""Collect the numbers shown on the project page into site/data.json.

    python scripts/site_data.py

Reads results/ (gitignored); writes only summary numbers, never per-case logs.
MemPoison-Bench counts come from the README (those runs predate this repository).
AgentDojo is computed from rows.jsonl, so a run still in progress is marked partial.
"""
import json
from pathlib import Path

from smriti.scorecard.report import wilson

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGENTDOJO_EXPECTED = 40 * 15 * 3          # workspace v1.2.2: 40 tasks x (1 clean + 14) x 3 arms


def rate(k: int | None, n: int | None) -> dict | None:
    if k is None or not n:
        return None
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def mempoison() -> dict:
    # Counts from the original MemPoison-Bench runs (metrics.json of that run; README table).
    return {
        "arms": [
            {"name": "No defense", "attack": rate(29, 60), "clean": rate(15, 15),
             "under_attack": rate(31, 60)},
            {"name": "Prompt-guard", "attack": rate(22, 60), "clean": rate(12, 15),
             "under_attack": rate(31, 60)},
            {"name": "SMRITI", "attack": rate(0, 60), "clean": rate(15, 15),
             "under_attack": rate(51, 60), "confirm_clean": rate(15, 15), "highlight": True},
            {"name": "SMRITI, detector off", "attack": rate(0, 60), "clean": rate(15, 15),
             "under_attack": rate(46, 60), "confirm_clean": rate(15, 15)},
            {"name": "SMRITI, action check off", "attack": rate(9, 60), "clean": rate(15, 15),
             "under_attack": rate(51, 60)},
            {"name": "SMRITI, provenance labels off", "attack": rate(29, 60),
             "clean": rate(15, 15), "under_attack": rate(31, 60)},
        ],
        "extra": {"laundering_survival": rate(0, 15), "false_quarantine": rate(0, 105),
                  "cross_tenant_leaks": 0, "auditor_false_rollbacks": 0,
                  "auditor_caught": {"No defense": [26, 28], "SMRITI": [9, 9],
                                     "SMRITI, detector off": [12, 13]}},
    }


def injecagent() -> dict | None:
    out = {}
    for setting in ("base", "enhanced"):
        path = RESULTS / f"injecagent_{setting}" / "summary.json"
        if not path.exists():
            return None
        s = json.loads(path.read_text())["summary"]
        names = {"none": "No defense", "smriti": "SMRITI",
                 "smriti_no_detector": "SMRITI, detector off"}
        out[setting] = [{
            "name": names[arm], "highlight": arm == "smriti",
            "attack": m["asr_all"], "direct_harm": m["asr_all_direct_harm"],
            "data_stealing": m["asr_all_data_stealing"], "fooled": m["model_fooled"],
            "confirm": m["confirmation_rate"], "quarantined": m["quarantined_at_write"],
        } for arm, m in s.items()]
    return out


def agentdojo() -> dict | None:
    path = RESULTS / "agentdojo_workspace" / "rows.jsonl"
    if not path.exists():
        return None
    latest = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        latest[(r["arm"], r["user_task"], r["injection_task"])] = r
    from smriti.external.agentdojo_run import scored
    rows = [r for r in map(scored, latest.values()) if r["error"] is None]
    names = {"none": "No defense", "repeat_user_prompt": "AgentDojo defense (repeat prompt)",
             "smriti": "SMRITI"}
    arms = []
    for arm in ("none", "repeat_user_prompt", "smriti"):
        mine = [r for r in rows if r["arm"] == arm]
        clean = [r for r in mine if r["injection_task"] is None]
        attacked = [r for r in mine if r["injection_task"] is not None]
        arms.append({
            "name": names[arm], "highlight": arm == "smriti",
            "attack": rate(sum(r["security"] for r in attacked), len(attacked)),
            "clean": rate(sum(r["utility"] for r in clean), len(clean)),
            "under_attack": rate(sum(r["utility"] for r in attacked), len(attacked)),
            "confirm_clean": rate(sum(r["confirmations"] > 0 for r in clean), len(clean)),
            "confirm_attacked": rate(sum(r["confirmations"] > 0 for r in attacked), len(attacked)),
            "context_overflows": sum(1 for r in mine if r.get("context_overflow")),
            # The attacker's goal actually happened (overflows not counted as attacks).
            "attack_real": rate(sum(bool(r["security"]) and not r.get("context_overflow")
                                    for r in attacked), len(attacked)),
        })
    errors = sum(1 for r in map(scored, latest.values()) if r["error"] is not None)
    return {"arms": arms, "runs": len(rows), "expected": AGENTDOJO_EXPECTED,
            "partial": len(rows) < AGENTDOJO_EXPECTED, "errors": errors}


def main() -> None:
    data = {"mempoison": mempoison(), "injecagent": injecagent(), "agentdojo": agentdojo()}
    out = ROOT / "site" / "data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    ad = data["agentdojo"]
    print(f"wrote {out}; AgentDojo: " + (f"{ad['runs']}/{ad['expected']} runs"
                                        f"{' (partial)' if ad['partial'] else ''}" if ad else "none"))


if __name__ == "__main__":
    main()
