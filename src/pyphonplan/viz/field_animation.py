"""Field activation animation.

Animate the 1D activation profile of a dynamic neural field over time,
showing how peaks emerge and compete frame by frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_field(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    threshold: float = 0.0,
    save_path: str | Path | None = None,
    fps: int = 30,
    show: bool = True,
) -> FuncAnimation:
    """Animate the 1D activation profile of a dynamic field over time.

    Parameters
    ----------
    t : np.ndarray
        Time array, shape (n_time,).
    x : np.ndarray
        Spatial (parameter) array, shape (n_x,).
    u : np.ndarray
        Activation array, shape (n_x, n_time).
    threshold : float
        Threshold value shown as a horizontal dotted line.
    save_path : str, Path, or None
        If provided, save the animation to this file. Extension determines
        format: '.mp4' uses ffmpeg, '.gif' uses pillow.
    fps : int
        Frames per second for saved animation.
    show : bool
        If True, call plt.show() (blocking).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Fixed y-limits across all frames
    y_min = float(u.min())
    y_max = float(u.max())
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(x.min(), x.max())

    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Activation")
    title = ax.set_title(f"t = {t[0]:.1f}")

    (line,) = ax.plot(x, u[:, 0], color="black", linewidth=1.5)

    def _update(frame: int):
        line.set_ydata(u[:, frame])
        title.set_text(f"t = {t[frame]:.1f}")
        return line, title

    anim = FuncAnimation(
        fig, _update, frames=u.shape[1], interval=1000 / fps, blit=False
    )

    if save_path is not None:
        save_path = Path(save_path)
        ext = save_path.suffix.lower()
        if ext == ".gif":
            anim.save(str(save_path), writer="pillow", fps=fps)
        else:
            anim.save(str(save_path), writer="ffmpeg", fps=fps)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim
