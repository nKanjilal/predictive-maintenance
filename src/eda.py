"""Step 1: Exploratory analysis. Saves charts to reports/figures/ and prints key stats."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import ROOT, FAILURE_MODES, load_dataset

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def rate_by_bin(df, col, bins):
    g = df.groupby(pd.cut(df[col], bins), observed=True)["failure"]
    out = pd.DataFrame({"rate": g.mean() * 100, "n": g.size()})
    out.index = [f"{max(int(i.left), 0)}–{int(i.right)}" for i in out.index]
    return out


def bar(ax, s, title, xlabel):
    ax.bar(s.index, s["rate"], color=BLUE, width=0.7)
    for i, (r, n) in enumerate(zip(s["rate"], s["n"])):
        ax.text(i, r + 0.4, f"{r:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Failure rate (%)")
    ax.tick_params(axis="x", rotation=30)


def main():
    df, y, modes = load_dataset()
    print(f"Rows: {len(df):,}  Failures: {y.sum()} ({y.mean():.2%})")
    print("Failure modes:", modes.sum().to_dict())
    print(df[["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]]
          .describe().round(1).to_string())

    # Failure rate vs tool wear and torque
    wear = rate_by_bin(df, "tool_wear_min", [-1, 50, 100, 150, 200, 220, 260])
    torque = rate_by_bin(df, "torque_nm", [0, 20, 30, 40, 50, 60, 80])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bar(axes[0], wear, "Failure rate climbs sharply past ~200 min of tool wear", "Tool wear (min)")
    bar(axes[1], torque, "Failures cluster at very low and very high torque", "Torque (Nm)")
    fig.tight_layout()
    fig.savefig(FIG / "01_failure_rate_wear_torque.png", bbox_inches="tight")
    print("\nFailure rate by tool wear:\n", wear.round(2).to_string())
    print("\nFailure rate by torque:\n", torque.round(2).to_string())

    # Interaction: torque x tool wear heatmap
    t_bins = [0, 20, 30, 40, 50, 60, 80]
    w_bins = [-1, 50, 100, 150, 200, 260]
    pivot = (df.assign(t=pd.cut(df.torque_nm, t_bins), w=pd.cut(df.tool_wear_min, w_bins))
             .pivot_table(index="t", columns="w", values="failure", aggfunc="mean", observed=False) * 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(pivot.shape[1]), [f"{int(c.left)+ (c.left<0)}–{int(c.right)}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{int(c.left)}–{int(c.right)}" for c in pivot.index])
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if v > 45 else INK)
    ax.set_xlabel("Tool wear (min)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("High torque + high tool wear = highest failure rate", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, label="Failure rate (%)")
    fig.tight_layout()
    fig.savefig(FIG / "02_torque_x_wear_heatmap.png", bbox_inches="tight")

    # Failure modes
    counts = modes.sum().reindex(FAILURE_MODES)
    names = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power",
             "OSF": "Overstrain", "RNF": "Random"}
    counts.index = [names[m] for m in counts.index]
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 1.5, i, str(v), va="center", fontsize=8)
    ax.set_title("Failure modes (a failure can have more than one)", loc="left", fontsize=11)
    ax.set_xlabel("Count")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "03_failure_modes.png", bbox_inches="tight")
    print("\nSaved figures to", FIG)


if __name__ == "__main__":
    main()
