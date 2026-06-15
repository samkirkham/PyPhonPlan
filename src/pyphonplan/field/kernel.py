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

    The kernel is sampled at the same spacing as the field (``x[1] - x[0]``),
    so that ``dx * convolve(g, kernel)`` is a consistent Riemann approximation
    of the interaction integral. Uses centre-based expansion so it works
    correctly for non-symmetric ranges (e.g. [100, 500] rather than [-10, 10]).

    Parameters
    ----------
    x : np.ndarray
        Original spatial array of the field (regularly spaced).
    expand : float
        Expansion factor for kernel range.

    Returns
    -------
    np.ndarray
        Spatial array centred on the field, spaced at the field's dx and
        spanning +/- expand * half-width (odd length).
    """
    if x.size < 2 or expand <= 0:
        raise ValueError("make_kernel_x requires len(x) >= 2 and expand > 0.")
    dx = x[1] - x[0]
    x_center = (x.max() + x.min()) / 2.0
    half_width_expanded = (x.max() - x.min()) / 2.0 * expand
    n_half = int(np.ceil(half_width_expanded / dx))
    return x_center + dx * np.arange(-n_half, n_half + 1)


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
        Excitatory strength (integrated area of the excitatory Gaussian).
    c_inh : float
        Inhibitory strength (integrated area of the inhibitory Gaussian).
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
    excite = (c_exc / np.sqrt(2 * np.pi * sigma_exc**2)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma_exc**2)
    )
    inhibit = (c_inh / np.sqrt(2 * np.pi * sigma_inh**2)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma_inh**2)
    )
    return excite - inhibit - c_global


def convolve(signal: np.ndarray, kernel: np.ndarray, fft: bool = True) -> np.ndarray:
    """Convolve signal with kernel, returning output of len(signal).

    Uses scipy.signal.fftconvolve by default for speed. Both paths return the
    window centred on the signal (the kernel may be longer than the signal).

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
        # fftconvolve 'same' is centred on the first argument (the signal).
        return fftconvolve(signal, kernel, mode="same")
    # np.convolve 'same' centres on the longer array, so take the signal-centred
    # window from the full convolution to match the fft path.
    c = np.convolve(signal, kernel, mode="full")
    start = (len(c) - len(signal)) // 2
    return c[start : start + len(signal)]
