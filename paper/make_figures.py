"""Generate paper figures for PyPhonPlan Interspeech paper.

Figure 1 (Section 2): 2-layer planning + memory heatmaps
Figure 2 (Section 3): Shadowing results — peak position bar chart + TD trajectories

Usage:
    uv run python paper/make_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyphonplan import FieldSystem, Targets, TaskDynamics
from pyphonplan.viz import plot_field_heatmap

# --- Matplotlib defaults for Interspeech ---
COLUMN_WIDTH = 3.35  # inches (single column)
FULL_WIDTH = 7.0     # inches (two-column)
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
})

OUTDIR = Path(__file__).parent / "figs"
OUTDIR.mkdir(exist_ok=True)


# ============================================================
# Figure 1: 2-layer planning + memory heatmaps
# ============================================================

def make_fig1():
    sys = FieldSystem(x_min=-10, x_max=10, step_size=0.02)

    kernel_params = dict(
        c_exc=4, c_inh=1, c_global=0.5,
        sigma_exc=1.0, sigma_inh=2.0, expand=3.0,
    )
    sys.add_field("planning", tau=25, h=-2, kernel_params=kernel_params)

    kernel_memory = dict(
        c_exc=4, c_inh=1, c_global=0.0,
        sigma_exc=1.0, sigma_inh=2.0, expand=3.0,
    )
    sys.add_field(
        "memory", tau=150, h=0, kernel_params=kernel_memory,
        field_type="memory", tau_decay=500, source_field="planning",
    )
    sys.add_coupling("memory", "planning", weight=2.0)

    sys.add_input(
        "planning", "input1",
        amplitude=10, position=-5, width=1.0, start=50, end=150,
    )
    sys.add_input(
        "planning", "input2",
        amplitude=10, position=5, width=1.0, start=100, end=200,
    )
    sys.solve(t_start=0, t_end=250)

    for name, title, fname in [
        ("planning", "Planning field", "fig1a.pdf"),
        ("memory", "Memory field", "fig1b.pdf"),
    ]:
        fig = plot_field_heatmap(
            sys.time, sys.x, sys.activation[name],
            title=title,
            figsize=(COLUMN_WIDTH, 2.2),
            show=False,
            activation_time=(name != "memory"),
        )
        ax = fig.axes[0]
        ax.get_legend().remove()
        ax.set_xlabel("Time (ms)")
        fig.savefig(OUTDIR / fname)
        plt.close(fig)
        print(f"  Saved {OUTDIR / fname}")


# ============================================================
# Figure 2: Shadowing results — bar chart + TD trajectories
# ============================================================

def make_fig2():
    # --- Simulation setup (matches shadowing.ipynb) ---
    sys = FieldSystem(x_min=-10, x_max=10, step_size=0.05)

    kp = dict(c_exc=4, c_inh=1, c_global=0.5, sigma_exc=1.0, sigma_inh=2.0)
    sys.add_field("planning", tau=25, h=-2, kernel_params=kp, gamma_gated=True)

    km = dict(c_exc=4, c_inh=1, c_global=0.0, sigma_exc=1.0, sigma_inh=2.0)
    sys.add_field(
        "memory", tau=150, h=0, kernel_params=km,
        field_type="memory", tau_decay=1000, source_field="planning",
    )

    kperc = dict(c_exc=4, c_inh=1, c_global=0.5, sigma_exc=1.0, sigma_inh=2.0)
    sys.add_field("perception", tau=10, h=-2, kernel_params=kperc)

    sys.add_coupling("memory", "planning", weight=10.0)
    sys.add_coupling("perception", "planning", weight=10.0, sigmoid=False)

    RESPONSE_POS = 3.0
    PERCEPTION_POS = 1.0
    N_SHADOW = 10

    def run_trial(has_perception=False, label=""):
        sys.clear_inputs()
        sys.add_input(
            "planning", "response",
            amplitude=30, position=RESPONSE_POS, width=1,
            start=100, end=200,
        )
        if has_perception:
            sys.add_input(
                "perception", "auditory",
                amplitude=40, position=PERCEPTION_POS, width=1,
                start=0, end=100,
            )
        sys.solve(t_start=0, t_end=300)
        return {
            "label": label,
            "planning": sys.activation["planning"].copy(),
            "time": sys.time.copy(),
        }

    # --- Run trials ---
    trials = [run_trial(has_perception=False, label="baseline")]
    for i in range(N_SHADOW):
        trials.append(run_trial(has_perception=True, label=f"shadow_{i+1}"))
    trials.append(run_trial(has_perception=False, label="washout"))

    # --- Measure plateau peaks ---
    def measure_plateau_peak(trial, t_lo=120, t_hi=180):
        targets = Targets(sys, activation=trial["planning"])
        pp, ap, t = targets.peak_activation(above_threshold=False)
        mask = (t >= t_lo) & (t <= t_hi) & (ap >= 0)
        if not np.any(mask):
            return np.nan
        return np.mean(pp[mask])

    plateau_peaks = [measure_plateau_peak(t) for t in trials]
    labels = [t["label"] for t in trials]
    bl_plateau = plateau_peaks[0]
    wo_plateau = plateau_peaks[-1]

    # --- TD with fixed targets ---
    GESTURE_DUR = 0.15
    td_baseline = TaskDynamics(t_start=0, t_end=GESTURE_DUR, initial_position=0.0)
    td_baseline.add_gesture(target=bl_plateau, k=8000, start=0.0, end=GESTURE_DUR)
    td_baseline.solve()

    td_washout = TaskDynamics(t_start=0, t_end=GESTURE_DUR, initial_position=0.0)
    td_washout.add_gesture(target=wo_plateau, k=8000, start=0.0, end=GESTURE_DUR)
    td_washout.solve()

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))

    # Panel A: Peak position bar chart
    ax = axes[0]
    colours = ["steelblue"] + ["coral"] * N_SHADOW + ["seagreen"]
    short_labels = (
        ["BL"]
        + [f"S{i+1}" for i in range(N_SHADOW)]
        + ["WO"]
    )
    ax.bar(range(len(plateau_peaks)), plateau_peaks, color=colours, width=0.7)
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=6)
    ax.set_ylabel("Peak position")
    ax.set_title("Planning peak position")
    ax.axhline(
        RESPONSE_POS, color="steelblue", lw=0.8, ls="--",
        label=f"Response ({RESPONSE_POS})",
    )
    ax.axhline(
        PERCEPTION_POS, color="coral", lw=0.8, ls=":",
        label=f"Perception ({PERCEPTION_POS})",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92))

    # Panel B: TD trajectories
    ax = axes[1]
    ax.plot(
        td_baseline.time * 1000, td_baseline.position,
        label="Baseline", color="steelblue",
    )
    ax.plot(
        td_washout.time * 1000, td_washout.position,
        label="Washout", color="seagreen",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Tract variable")
    ax.set_title("Tract variable trajectories")
    ax.legend()

    fig.savefig(OUTDIR / "fig2.pdf")
    plt.close(fig)
    print(f"  Saved {OUTDIR / 'fig2.pdf'}")

    # Print summary
    gap = RESPONSE_POS - PERCEPTION_POS
    shift = wo_plateau - bl_plateau
    print(f"  Peak shift: {shift:.3f} ({abs(shift)/gap*100:.1f}% of {gap:.0f}-unit gap)")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
