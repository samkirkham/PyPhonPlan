"""Interaction kernel and utility functions for dynamic fields.
"""

import numpy as np
from scipy.signal import fftconvolve


def sigmoid(x: np.ndarray, beta: float = 1.5, threshold: float = 0.0) -> np.ndarray:
    """Sigmoidal thresholding function."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-beta * (x - threshold)))


def make_kernel_x(x: np.ndarray, expand: float = 3.0) -> np.ndarray:
    """Create expanded spatial array for kernel computation.

    Uses centre-based expansion so it works correctly for non-symmetric
    ranges (e.g. [100, 500] rather than [-10, 10]).

    Parameters
    ----------
    x : np.ndarray
        Original spatial array of the field.
    expand : float
        Expansion factor for kernel range.

    Returns
    -------
    np.ndarray
        Expanded spatial array with int(len(x) * expand) points.
    """
    x_center = (x.max() + x.min()) / 2.0
    half_width = (x.max() - x.min()) / 2.0
    half_width_expanded = half_width * expand
    return np.linspace(
        x_center - half_width_expanded,
        x_center + half_width_expanded,
        int(len(x) * expand),
    )


def interaction_kernel(
    x: np.ndarray,
    c_exc: float,
    c_inh: float,
    c_global: float,
    sigma_exc: float,
    sigma_inh: float,
    mu: float = 0.0,
) -> np.ndarray:
    """Difference-of-Gaussians interaction kernel.

    Parameters
    ----------
    x : np.ndarray
        Spatial array (typically from make_kernel_x).
    c_exc : float
        Excitatory strength.
    c_inh : float
        Inhibitory strength.
    c_global : float
        Global inhibition constant.
    sigma_exc : float
        Excitatory width (narrow).
    sigma_inh : float
        Inhibitory width (broad).
    mu : float
        Centre of the kernel.

    Returns
    -------
    np.ndarray
        Kernel values: excitation - inhibition - global.
    """
    excite = (c_exc / np.sqrt(2 * np.pi * sigma_exc)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma_exc**2)
    )
    inhibit = (c_inh / np.sqrt(2 * np.pi * sigma_inh)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma_inh**2)
    )
    return excite - inhibit - c_global


def convolve(signal: np.ndarray, kernel: np.ndarray, fft: bool = True) -> np.ndarray:
    """Convolve signal with kernel, returning output of len(signal).

    Uses scipy.signal.fftconvolve by default for speed.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (typically sigmoid(u)).
    kernel : np.ndarray
        Interaction kernel.
    fft : bool
        If True (default), use FFT-based convolution.
    """
    if fft:
        c = fftconvolve(signal, kernel, mode="same")
    else:
        c = np.convolve(signal, kernel, mode="same")
    if len(c) == len(signal):
        return c
    else:
        diff = len(c) - len(signal)
        trim = diff // 2
        return c[trim : trim + len(signal)]
