"""Task dynamics visualisation functions."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_trajectory(
    time: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray | None = None,
    abs_velocity: bool = False,
    show: bool = True,
) -> Figure:
    """Plot articulatory trajectory from task dynamics.

    Parameters
    ----------
    time : np.ndarray
        Time array in seconds.
    position : np.ndarray
        Position trajectory.
    velocity : np.ndarray or None
        If provided, plot velocity on secondary axis.
    """
    if velocity is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        ax1.plot(time, position)
        ax1.set_ylabel("Position")
        vel = np.abs(velocity) if abs_velocity else velocity
        ax2.plot(time, vel, color="tab:orange")
        ax2.set_ylabel("|Velocity|" if abs_velocity else "Velocity")
        ax2.set_xlabel("Time (s)")
    else:
        fig, ax1 = plt.subplots()
        ax1.plot(time, position)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position")

    if show:
        plt.tight_layout()
        plt.show()
    return fig


def plot_blended_params(
    time: np.ndarray,
    blended_k: np.ndarray,
    blended_target: np.ndarray,
    blended_damping: np.ndarray,
    params: list[str] | None = None,
    show: bool = True,
) -> Figure:
    """Plot blended task-dynamic parameters over time.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    blended_k, blended_target, blended_damping : np.ndarray
        Per-timestep blended parameter arrays.
    params : list of str or None
        Subset of ['target', 'k', 'damping'] to plot. None plots all three.
    """
    all_panels = [
        ("target", blended_target, "Target"),
        ("k", blended_k, "Stiffness"),
        ("damping", blended_damping, "Damping"),
    ]
    if params is not None:
        valid = {key for key, _, _ in all_panels}
        bad = set(params) - valid
        if bad:
            raise ValueError(f"Unknown param keys: {bad}. Valid: {valid}")
        all_panels = [(k, d, l) for k, d, l in all_panels if k in params]

    n = len(all_panels)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8, 2 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, (_, data, label) in zip(axes, all_panels):
        ax.plot(time, data)
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Time (s)")

    if show:
        plt.tight_layout()
        plt.show()
    return fig
