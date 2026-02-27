"""Single dynamic neural field simulation.

Implements the Amari equation for a 1D dynamic field:
    tau * du/dt = -u + h + s(t) + conv(sigmoid(u), kernel) + noise

Based on dynamicfields.py and phonology_wo_symbols/figures/dnf/functions.py.
"""

import numpy as np
from matplotlib.figure import Figure

from pyphonplan.field.inputs import GaussianInput
from pyphonplan.field.kernel import interaction_kernel, sigmoid, convolve, make_kernel_x


class DynamicField:
    """Single dynamic neural field.

    Parameters
    ----------
    x_min, x_max : float
        Spatial range of the field.
    step_size : float
        Spatial resolution.
    """

    def __init__(self, x_min: float = -10.0, x_max: float = 10.0, step_size: float = 0.1):
        self.x = np.arange(x_min, x_max + step_size / 2, step_size)
        self._dx = step_size
        self._inputs: dict[str, GaussianInput] = {}
        self._sigmoid_beta = 1.5
        self._sigmoid_threshold = 0.0
        self.kernel: np.ndarray | None = None
        self.time: np.ndarray | None = None
        self.activation: np.ndarray | None = None

    def set_sigmoid(self, beta: float = 1.5, threshold: float = 0.0):
        """Set sigmoid parameters for the thresholding function."""
        self._sigmoid_beta = beta
        self._sigmoid_threshold = threshold

    def set_kernel(
        self,
        c_exc: float,
        c_inh: float,
        c_global: float,
        sigma_exc: float,
        sigma_inh: float,
        mu: float = 0.0,
        expand: float = 3.0,
    ):
        """Build the difference-of-Gaussians interaction kernel."""
        kernel_x = make_kernel_x(self.x, expand)
        self.kernel = interaction_kernel(kernel_x, c_exc, c_inh, c_global, sigma_exc, sigma_inh, mu)

    def add_input(
        self,
        name: str,
        amplitude: float,
        position: float,
        width: float,
        offset: float = 0.0,
        start: int = 0,
        end: int = 0,
    ) -> GaussianInput:
        """Add a named Gaussian input with timing."""
        inp = GaussianInput(
            name=name,
            amplitude=amplitude,
            position=position,
            width=width,
            offset=offset,
            start=start,
            end=end,
        )
        self._inputs[name] = inp
        return inp

    def plot_sigmoid(self, show: bool = True) -> Figure:
        """Plot the current sigmoid thresholding function."""
        from pyphonplan.viz.field_plots import plot_sigmoid
        return plot_sigmoid(self.x, self._sigmoid_beta, self._sigmoid_threshold, show=show)

    def plot_kernel(self, show: bool = True) -> Figure:
        """Plot the current interaction kernel."""
        if self.kernel is None:
            raise RuntimeError("Call set_kernel() before plot_kernel().")
        from pyphonplan.viz.field_plots import plot_kernel
        return plot_kernel(self.x, self.kernel, show=show)

    def plot_inputs(self, show: bool = True) -> Figure:
        """Plot all Gaussian input profiles."""
        if not self._inputs:
            raise RuntimeError("No inputs added yet.")
        from pyphonplan.viz.field_plots import plot_inputs
        return plot_inputs(self.x, self._inputs, show=show)

    def _sum_inputs_at(self, t: int) -> np.ndarray:
        """Sum all active inputs at time step t."""
        total = np.zeros_like(self.x)
        for inp in self._inputs.values():
            if inp.is_active(t):
                total += inp.evaluate(self.x)
        return total

    def solve(
        self,
        t_start: int,
        t_end: int,
        dt: int = 1,
        y0: np.ndarray | None = None,
        tau: float = 50.0,
        h: float = -2.0,
        noise: float = 0.0,
    ):
        """Solve the dynamic field equation using Euler-Maruyama integration.

        After solving, `self.time` and `self.activation` (shape: n_x x n_time)
        are set.

        Parameters
        ----------
        t_start, t_end : int
            Time range (integer time steps).
        dt : int
            Time step.
        y0 : np.ndarray or None
            Initial field state. If None, uses h * np.ones(len(x)).
        tau : float
            Time constant.
        h : float
            Resting level.
        noise : float
            Noise amplitude (0 = deterministic).
        """
        if self.kernel is None:
            raise RuntimeError("Call set_kernel() before solve().")

        if y0 is None:
            y0 = h * np.ones(len(self.x))

        self.time = np.arange(t_start, t_end + dt, dt)
        n_steps = len(self.time)
        n_x = len(self.x)
        activation = np.empty((n_x, n_steps))
        u = y0.copy()
        activation[:, 0] = u
        noise_scale = noise * np.sqrt(dt) / tau if noise > 0 else 0.0

        for i in range(1, n_steps):
            t = self.time[i - 1]
            s = self._sum_inputs_at(round(t))
            dudt = (
                -u + h + s
                + self._dx * convolve(sigmoid(u, self._sigmoid_beta, self._sigmoid_threshold), self.kernel)
            ) / tau
            u = u + dt * dudt
            if noise_scale > 0:
                u += noise_scale * np.random.normal(0, 1, n_x)
            activation[:, i] = u

        self.activation = activation
