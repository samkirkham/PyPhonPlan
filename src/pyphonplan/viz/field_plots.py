"""Field visualisation functions.

Based on plot_field_heatmap and plot_field from
phonology_wo_symbols/figures/dnf/functions.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch, Rectangle

from pyphonplan.field.kernel import sigmoid


def plot_field_heatmap(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    threshold: float = 0.0,
    title: str | None = None,
    show: bool = True,
    inset: bool = False,
    colorbar: bool = True,
    activation_time: bool = True,
    time_step: int | None = None,
    suppress_labels: bool = False,
    inset_box: bool = False,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """2D heatmap of dynamic field activation over time and parameter.

    Publication-ready plot with optional peak trajectory overlay,
    threshold crossing line, and activation profile inset.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Spatial (parameter) array.
    u : np.ndarray
        Activation array, shape (n_x, n_time).
    threshold : float
        Threshold for crossing detection.
    title : str or None
        Plot title.
    show : bool
        If True, call plt.show().
    inset : bool
        If True, add inset showing activation profile at time_step.
    colorbar : bool
        If True, show colorbar.
    activation_time : bool
        If True, show threshold crossing line.
    time_step : int or None
        Time index for inset/box. If None, uses final time step.
    suppress_labels : bool
        If True, remove labels and ticks (for grid assembly).
    inset_box : bool
        If True, draw box around peak at time_step.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    extent = (t.min(), t.max(), x.min(), x.max())
    im = ax.imshow(u, aspect="auto", extent=extent, origin="lower", cmap="viridis")

    if colorbar:
        plt.colorbar(im, ax=ax, label="Activation")

    if not suppress_labels:
        ax.set_xlabel("Time")
        ax.set_ylabel("Parameter")
        if title:
            ax.set_title(title)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Threshold crossing line
    crossing_idx = None
    if activation_time:
        above_any = np.any(u >= threshold, axis=0)
        if np.any(above_any):
            crossing_idx = int(np.argmax(above_any))
        if crossing_idx is not None:
            ax.axvline(
                x=t[crossing_idx], color="red", linewidth=2,
                label=f"Activation >= {threshold} at t={t[crossing_idx]:.2f}",
            )

    # Peak activation trajectory: only show when there is a focused peak
    # (peak must exceed threshold AND stand out from the field minimum)
    peak_indices = np.argmax(u, axis=0)
    peak_locations = x[peak_indices]
    peak_activation_values = u[peak_indices, np.arange(u.shape[1])]
    min_activation_values = np.min(u, axis=0)
    contrast = peak_activation_values - min_activation_values
    has_peak = (peak_activation_values > threshold) & (contrast > 1.0)
    peaks_above = np.where(has_peak, peak_locations, np.nan)
    if crossing_idx is not None:
        peaks_above[:crossing_idx] = np.nan
    ax.plot(t, peaks_above, linestyle="--", color="white", linewidth=2, label="Peak activation")

    if inset:
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper right")

    # Box around peak at specified time step
    if inset_box:
        ts = time_step if time_step is not None else -1
        target_t = t[ts]
        target_u = u[:, ts]
        peak_x = x[np.argmax(target_u)]
        window_width = 0.5
        time_window = 0.05 * (t.max() - t.min())

        rect = Rectangle(
            (target_t - time_window, peak_x - window_width),
            2 * time_window,
            2 * window_width,
            linewidth=2, edgecolor="white", facecolor="none",
        )
        ax.add_patch(rect)

    # Inset: activation profile at time step
    if inset:
        ts = time_step if time_step is not None else -1
        target_t = t[ts]
        target_u = u[:, ts]
        peak_x = x[np.argmax(target_u)]
        window_width = 0.5
        rect_x_min = target_t - 0.1 * (t.max() - t.min())
        rect_x_max = target_t

        rect = Rectangle(
            (rect_x_min, peak_x - window_width),
            rect_x_max - rect_x_min,
            2 * window_width,
            linewidth=2, edgecolor="gray", facecolor="none",
        )
        ax.add_patch(rect)

        inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")
        inset_ax.axvline(peak_x, color="gray", linestyle="-", linewidth=1)
        inset_ax.axhline(0, color="red", linestyle="--", linewidth=1)
        inset_ax.plot(x, target_u, color="black")
        inset_ax.set_xlabel("Parameter")
        inset_ax.set_ylabel("Activation")
        inset_ax.tick_params(axis="both", colors="white")
        inset_ax.xaxis.label.set_color("white")
        inset_ax.yaxis.label.set_color("white")

        arrow = FancyArrowPatch(
            posA=((rect_x_min + rect_x_max) / 2, (peak_x - window_width + peak_x + window_width) / 2),
            posB=(t.max() - 0.05 * (t.max() - t.min()), peak_x + 2.5),
            arrowstyle="->", mutation_scale=15, color="white", linewidth=2,
        )
        ax.add_artist(arrow)

    if show:
        plt.tight_layout(pad=0.2)
        plt.show()

    return fig


def plot_field_surface(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    axis_labels: list[str] | None = None,
    zlim: list[float] | None = None,
    threshold: float | None = None,
) -> Figure:
    """3D surface plot of field activation.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Spatial (parameter) array.
    u : np.ndarray
        Activation array, shape (n_x, n_time).
    axis_labels : list[str] or None
        Labels for [time, parameter, activation] axes.
    zlim : list[float] or None
        Z-axis limits.
    threshold : float or None
        If provided, draw a semi-transparent plane at this z-value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if axis_labels is None:
        axis_labels = ["Time", "Parameter", "Activation"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    T, X = np.meshgrid(t, x)
    ax.plot_surface(T, X, u, cmap="viridis", alpha=0.9)
    if threshold is not None:
        Z = np.full_like(T, threshold)
        ax.plot_surface(T, X, Z, color="grey", alpha=0.3)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    if zlim is not None:
        ax.set_zlim(zlim)
    plt.show()
    return fig


def plot_inputs(
    x: np.ndarray,
    inputs: dict,
    show: bool = True,
) -> Figure:
    """Plot Gaussian input profiles.

    Parameters
    ----------
    x : np.ndarray
        Spatial array.
    inputs : dict
        Mapping of name -> GaussianInput.
    """
    fig, ax = plt.subplots()
    for name, inp in inputs.items():
        ax.plot(x, inp.evaluate(x), label=f"{name} (t={inp.start}-{inp.end})")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Input strength")
    ax.legend()
    if show:
        plt.show()
    return fig


def plot_kernel(
    x: np.ndarray,
    kernel: np.ndarray,
    threshold: float = 0.0,
    show: bool = True,
) -> Figure:
    """Plot the interaction kernel (central portion)."""
    fig, ax = plt.subplots()
    n_x = len(x)
    mid = len(kernel) // 2
    start = mid - n_x // 2
    ax.plot(x, kernel[start:start + n_x])
    ax.axhline(y=threshold, color="lightgray", linestyle="--")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Kernel strength")
    if show:
        plt.show()
    return fig


def plot_sigmoid(
    x: np.ndarray,
    beta: float = 1.5,
    threshold: float = 0.0,
    show: bool = True,
) -> Figure:
    """Plot the sigmoid thresholding function."""
    fig, ax = plt.subplots()
    ax.plot(x, sigmoid(x, beta, threshold))
    ax.set_xlabel("x")
    ax.set_ylabel("sigmoid(x)")
    if show:
        plt.show()
    return fig
