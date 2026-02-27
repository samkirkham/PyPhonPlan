"""Task dynamics visualisation functions."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_trajectory(
    time: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray | None = None,
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
        ax2.plot(time, velocity, color="tab:orange")
        ax2.set_ylabel("Velocity")
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
    show: bool = True,
) -> Figure:
    """Plot blended task-dynamic parameters over time.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    blended_k, blended_target, blended_damping : np.ndarray
        Per-timestep blended parameter arrays.
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(time, blended_target)
    axes[0].set_ylabel("Target")

    axes[1].plot(time, blended_k)
    axes[1].set_ylabel("Stiffness")

    axes[2].plot(time, blended_damping)
    axes[2].set_ylabel("Damping")
    axes[2].set_xlabel("Time (s)")

    if show:
        plt.tight_layout()
        plt.show()
    return fig
