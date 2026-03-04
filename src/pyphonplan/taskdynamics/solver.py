"""Task dynamic solver (Saltzman & Munhall 1989).

Standalone solver for articulatory trajectories driven by gestural
specifications. Extracted from pygest, stripped of pandas dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

NEUTRAL_STIFFNESS = 2000.0


@dataclass
class Gesture:
    """A single gestural specification.

    Parameters
    ----------
    target : float
        Target position for this gesture.
    k : float
        Spring stiffness.
    damping : float or None
        Damping coefficient. If None, auto-set to 2*sqrt(k) (critical damping).
    start : float
        Onset time in seconds.
    end : float
        Offset time in seconds.
    alpha : float
        Blending weight (activation level).
    """

    target: float
    k: float
    damping: float | None = None
    start: float = 0.0
    end: float = 0.0
    alpha: float = 1.0

    def __post_init__(self):
        if self.damping is None:
            self.damping = 2.0 * np.sqrt(self.k)


def _sm89(t, state, k, b, target):
    """Saltzman & Munhall 1989 task dynamic equation.

    Second-order ODE: x_ddot = -b * x_dot - k * (x - target), with m=1.
    """
    x, v = state
    return [v, -b * v - k * (x - target)]


def _build_blended_params(
    gestures: list[Gesture],
    time: np.ndarray,
    neutral_target: float = 0.0,
    neutral_stiffness: float = NEUTRAL_STIFFNESS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-timestep blended stiffness, target, and damping arrays.

    When multiple gestures overlap, their parameters are blended via
    alpha-weighted averaging. When no gesture is active, neutral
    attractor parameters apply.
    """
    n = len(time)
    neutral_damping = 2.0 * np.sqrt(neutral_stiffness)

    blended_k = np.full(n, neutral_stiffness)
    blended_target = np.full(n, neutral_target)
    blended_damping = np.full(n, neutral_damping)

    weight_sum = np.zeros(n)
    k_sum = np.zeros(n)
    target_sum = np.zeros(n)
    damping_sum = np.zeros(n)

    for g in gestures:
        if g.alpha <= 0:
            continue
        active = (time >= g.start) & (time <= g.end)
        weight_sum[active] += g.alpha
        k_sum[active] += g.k * g.alpha
        target_sum[active] += g.target * g.alpha
        damping_sum[active] += g.damping * g.alpha  # type: ignore[operator]  # __post_init__ guarantees non-None

    active_mask = weight_sum > 0
    if np.any(active_mask):
        blended_k[active_mask] = k_sum[active_mask] / weight_sum[active_mask]
        blended_target[active_mask] = target_sum[active_mask] / weight_sum[active_mask]
        blended_damping[active_mask] = damping_sum[active_mask] / weight_sum[active_mask]

    return blended_k, blended_target, blended_damping


class TaskDynamics:
    """OO interface for task dynamic simulation.

    Usage::

        td = TaskDynamics(t_start=0.0, t_end=0.5, dt=0.001)
        td.add_gesture(target=5.0, k=100.0, start=0.05, end=0.25)
        td.solve()
        td.plot()
        td.plot_params()
    """

    def __init__(
        self,
        t_start: float = 0.0,
        t_end: float = 1.0,
        dt: float = 0.001,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        neutral_target: float = 0.0,
        neutral_stiffness: float = NEUTRAL_STIFFNESS,
        method: str = "LSODA",
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.neutral_target = neutral_target
        self.neutral_stiffness = neutral_stiffness
        self.method = method

        self._gestures: list[Gesture] = []

        # Populated by solve()
        self.time: np.ndarray | None = None
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.blended_k: np.ndarray | None = None
        self.blended_target: np.ndarray | None = None
        self.blended_damping: np.ndarray | None = None

    def add_gesture(
        self,
        target: float,
        k: float,
        damping: float | None = None,
        start: float = 0.0,
        end: float = 0.0,
        alpha: float = 1.0,
    ) -> Gesture:
        """Create and register a gesture."""
        g = Gesture(
            target=target,
            k=k,
            damping=damping,
            start=start,
            end=end,
            alpha=alpha,
        )
        self._gestures.append(g)
        return g

    def solve(self):
        """Compute blended parameters and solve the ODE."""
        time = np.arange(self.t_start, self.t_end + self.dt / 2, self.dt)
        blended_k, blended_target, blended_damping = _build_blended_params(
            self._gestures, time, self.neutral_target, self.neutral_stiffness
        )

        dt = self.dt
        t_start = self.t_start

        def _ode(t, state):
            idx = int((t - t_start) / dt)
            idx = max(0, min(idx, len(time) - 1))
            return _sm89(t, state, blended_k[idx], blended_damping[idx], blended_target[idx])

        sol = solve_ivp(
            _ode,
            [self.t_start, self.t_end],
            [self.initial_position, self.initial_velocity],
            method=self.method,
            t_eval=time,
            max_step=self.dt,
        )
        if not sol.success:
            raise RuntimeError(f"Task dynamics solve failed: {sol.message}")

        self.time = time
        self.position = sol.y[0]
        self.velocity = sol.y[1]
        self.blended_k = blended_k
        self.blended_target = blended_target
        self.blended_damping = blended_damping

    def solve_from_trace(self, time, target, k=None, damping=None, time_scale=0.001):
        """Solve task dynamics from a pre-computed target trace.

        Drives the task dynamic ODE with a time-varying target and constant
        spring parameters. Designed for ``above_threshold=True`` output from
        ``Targets.peak_activation()``: the active period defines the
        simulation window, with onset mapped to t=0.

        ODE duration is derived from the field's time span:
        ``duration = (time[-1] - time[0]) * time_scale``. The default
        ``time_scale=0.001`` converts field time steps (ms) to ODE
        seconds. Each field sample maps by index onto the ODE time grid
        — no interpolation is needed.

        Parameters
        ----------
        time : np.ndarray
            Time array from peak_activation (must be regularly spaced).
        target : np.ndarray
            Target position at each timestep. Same length as *time*.
        k : float or None
            Spring stiffness. Defaults to ``self.neutral_stiffness``.
        damping : float or None
            Damping coefficient. Defaults to ``2*sqrt(k)`` (critical damping).
        time_scale : float
            Multiplier converting field time units to ODE seconds.
            Default ``0.001`` (field ms → seconds).

        Raises
        ------
        ValueError
            If *time* contains temporal gaps (non-uniform spacing).
        """
        time = np.asarray(time, dtype=float)
        target = np.asarray(target, dtype=float)
        if time.shape != target.shape:
            raise ValueError("time and target must have the same length")

        if k is None:
            k = self.neutral_stiffness
        if damping is None:
            damping = 2.0 * np.sqrt(k)

        # Check for temporal gaps (catches mid-gesture threshold drops)
        diffs = np.diff(time)
        if not np.allclose(diffs, diffs[0]):
            raise ValueError(
                "time array has temporal gaps (non-uniform spacing). "
                "This likely means the input activation drops below "
                "threshold mid-gesture. Check the field activation."
            )

        # ODE duration derived from field time span
        duration = (time[-1] - time[0]) * time_scale
        n = len(target)
        time_solve = np.linspace(0, duration, n)
        dt = duration / (n - 1)
        t_end = time_solve[-1]

        blended_k = np.full(n, k)
        blended_damping = np.full(n, damping)

        def _ode(t, state):
            idx = int(t / dt)
            idx = max(0, min(idx, n - 1))
            return _sm89(t, state, k, damping, target[idx])

        sol = solve_ivp(
            _ode,
            [0.0, t_end],
            [self.initial_position, self.initial_velocity],
            method=self.method,
            t_eval=time_solve,
            max_step=dt,
        )
        if not sol.success:
            raise RuntimeError(f"Task dynamics solve failed: {sol.message}")

        self.time = time_solve
        self.position = sol.y[0]
        self.velocity = sol.y[1]
        self.blended_k = blended_k
        self.blended_target = target
        self.blended_damping = blended_damping

    def _check_solved(self):
        if self.time is None:
            raise RuntimeError("Call solve() before plotting.")

    def plot(self, abs_velocity: bool = False, show: bool = True):
        """Plot position and velocity trajectories.

        Parameters
        ----------
        abs_velocity : bool
            If True, plot absolute velocity instead of signed velocity.
        """
        self._check_solved()
        from pyphonplan.viz.task_plots import plot_trajectory
        return plot_trajectory(
            self.time, self.position, self.velocity,
            abs_velocity=abs_velocity, show=show,
        )

    def plot_params(self, params: list[str] | None = None, show: bool = True):
        """Plot blended parameters.

        Parameters
        ----------
        params : list of str or None
            Subset of ['k', 'target', 'damping'] to plot. None plots all.
        """
        self._check_solved()
        from pyphonplan.viz.task_plots import plot_blended_params
        return plot_blended_params(
            self.time,
            self.blended_k,
            self.blended_target,
            self.blended_damping,
            params=params,
            show=show,
        )
