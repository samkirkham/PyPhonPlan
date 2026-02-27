"""Target and blending visualisation functions."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_target_activations(
    time: np.ndarray,
    traces: list,
    threshold: float = 0.0,
    show: bool = True,
) -> Figure:
    """Plot activation time series at each target position.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    traces : list[TargetTrace]
        List of TargetTrace objects.
    threshold : float
        Threshold line to display.
    """
    fig, ax = plt.subplots()
    for trace in traces:
        ax.plot(time, trace.activation, label=f"x={trace.position:.2f}")
    ax.axhline(y=threshold, color="lightgray", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Activation")
    ax.legend()
    if show:
        plt.show()
    return fig


def plot_peak_activation(
    parameter_peak: np.ndarray,
    activation_peak: np.ndarray,
    time: np.ndarray,
    dimension: str = "parameter",
    show: bool = True,
) -> Figure:
    """Plot peak activation trajectory.

    Parameters
    ----------
    parameter_peak : np.ndarray
        x-position of peak activation per time step.
    activation_peak : np.ndarray
        Peak activation value per time step.
    time : np.ndarray
        Corresponding time values.
    dimension : str
        "parameter" to plot x-position, "activation" to plot activation value.
    """
    fig, ax = plt.subplots()
    if dimension == "parameter":
        ax.plot(time, parameter_peak)
        ax.set_ylabel("Parameter (peak)")
    elif dimension == "activation":
        ax.plot(time, activation_peak)
        ax.set_ylabel("Activation (peak)")
    else:
        raise ValueError("dimension must be 'parameter' or 'activation'.")
    ax.set_xlabel("Time")
    if show:
        plt.show()
    return fig
