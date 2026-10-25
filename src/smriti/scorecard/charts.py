"""PNG charts for the slides: ASR by arm x class, utility by arm, ablations.

Colors: reference categorical slots in fixed order, one per arm (identity follows the arm,
never its rank). Error bars are Wilson 95% intervals. Values are printed on bars because
slot 3 is below 3:1 contrast on the light surface (relief rule); scorecard.md is the table.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smriti.scorecard.report import ARM_LABEL, ATTACK_ORDER, wilson  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")          # categorical slots 1-3
ARM_COLOR = {"A0": SLOTS[0], "A1": SLOTS[1], "A2": SLOTS[2]}
SEQ = "#2a78d6"                                    # single series charts

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def _err(rates: list[dict]) -> list[list[float]]:
    lows, highs = [], []
    for r in rates:
        lo, hi = wilson(r["k"], r["n"]) if r and r["n"] else (0.0, 0.0)
        rate = r["rate"] if r and r["n"] else 0.0
        lows.append(max(0.0, rate - lo) * 100)
        highs.append(max(0.0, hi - rate) * 100)
    return [lows, highs]


def _bars(ax, xs, rates, width, color, label=None, hatch=None):
    colors = color if isinstance(color, list) else [color] * len(rates)
    heights = [100 * (r["rate"] or 0) if r and r["n"] else 0 for r in rates]
    err = _err(rates)
    bars = ax.bar(xs, heights, width * 0.92, color=colors, label=label, zorder=2,
                  edgecolor=SURFACE, linewidth=2)
    if hatch:  # texture: secondary encoding that survives grayscale and CVD
        bars = ax.bar(xs, heights, width * 0.92, color="none", hatch=hatch, zorder=2.5,
                      edgecolor=SURFACE, linewidth=0)
    ax.errorbar(xs, heights, yerr=err, fmt="none", ecolor=TEXT_2, elinewidth=1,
                capsize=3, zorder=3)
    for x, h, up, r in zip(xs, heights, err[1], rates):
        if r and r["n"]:  # above the whisker so the label never sits on the error bar
            ax.text(x, h + up + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=9,
                    color=TEXT_2, zorder=4)
    return bars


def _axes(ax, ylabel):
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0)


def asr_by_arm_class(metrics: dict, out: Path) -> Path:
    arms = [a for a in ("A0", "A1", "A2") if a in metrics["asr"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        xs = [g + (i - (len(arms) - 1) / 2) * width for g in range(len(ATTACK_ORDER))]
        _bars(ax, xs, [metrics["asr"][arm].get(a) for a in ATTACK_ORDER], width,
              ARM_COLOR[arm], ARM_LABEL[arm])
    ax.set_xticks(range(len(ATTACK_ORDER)), [a.capitalize() for a in ATTACK_ORDER])
    _axes(ax, "Attack success rate (%)")
    ax.set_title("Attack success by arm and attack class (lower is better)")
    ax.legend(frameon=False, loc="upper left", ncols=len(arms))
    return _save(fig, out / "asr_by_arm_class.png")


def utility_by_arm(metrics: dict, out: Path) -> Path:
    arms = [a for a in ("A0", "A1", "A2") if a in metrics["per_arm"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Color = arm (same as the ASR chart); texture = "under attack".
    kinds = (("utility_clean", "Clean episodes", None), ("utility_under_attack", "Under attack", "///"))
    width = 0.38
    colors = [ARM_COLOR[a] for a in arms]
    for i, (key, _, hatch) in enumerate(kinds):
        xs = [g + (i - 0.5) * width for g in range(len(arms))]
        _bars(ax, xs, [metrics["per_arm"][a][key] for a in arms], width, colors, hatch=hatch)
    ax.set_xticks(range(len(arms)), [ARM_LABEL[a] for a in arms])
    _axes(ax, "Legitimate task completed (%)")
    ax.set_title("Utility by arm (higher is better)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=TEXT_2, edgecolor=SURFACE, label="Clean episodes"),
                       Patch(facecolor=TEXT_2, edgecolor=SURFACE, hatch="///",
                             label="Under attack")],
              frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out / "utility_by_arm.png")


def ablations(metrics: dict, out: Path) -> Path | None:
    arms = [a for a in ("A2", "no_write_flagging", "no_read_escaping", "no_action_check",
                        "no_provenance") if a in metrics["asr"]]
    if len(arms) < 2:
        return None
    labels = ["A2 (all layers)" if a == "A2" else ARM_LABEL[a].replace("A2 - ", "minus ")
              for a in arms]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _bars(axes[0], range(len(arms)), [metrics["asr"][a]["all"] for a in arms], 0.6, SEQ)
    axes[0].set_title("Attack success (lower is better)")
    _bars(axes[1], range(len(arms)),
          [metrics["per_arm"][a]["utility_clean"] for a in arms], 0.6, SEQ)
    axes[1].set_title("Clean utility (higher is better)")
    for ax in axes:
        ax.set_xticks(range(len(arms)), labels, rotation=20, ha="right")
        _axes(ax, "%")
    fig.suptitle("Ablations: SMRITI with one layer removed", x=0.01, ha="left",
                 fontweight="bold")
    return _save(fig, out / "ablations.png")


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_charts(metrics: dict, out: Path) -> list[Path]:
    paths = [asr_by_arm_class(metrics, out), utility_by_arm(metrics, out),
             ablations(metrics, out)]
    return [p for p in paths if p is not None]
