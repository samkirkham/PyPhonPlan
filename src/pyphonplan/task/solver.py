"""Task dynamic solver (Saltzman & Munhall 1989).

Standalone solver for articulatory trajectories driven by gestural
specifications. Extracted from pygest, stripped of pandas dependencies.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


NEUTRAL_STIFFNESS = 4.0


@dataclass
class Gesture:
    """A single gestural specification.

    Parameters
    ----------
    target : float
        Target position for this gesture.
    stiffness : float
        Spring stiffness (k).
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
    stiffness: float
    damping: float | None = None
    start: float = 0.0
    end: float = 0.0
    alpha: float = 1.0

    def __post_init__(self):
        if self.damping is None:
            self.damping = 2.0 * np.sqrt(self.stiffness)


def _sm89(t, state, k, b, target):
    """Saltzman & Munhall 1989 task dynamic equation.

    Second-order ODE: x_ddot = -b * x_dot - k * (x - target), with m=1.
    """
    x, v = state
    return [v, -b * v - k * (x - target)]


def build_blended_params(
    gestures: list[Gesture],
    time: np.ndarray,
    neutral_target: float = 0.0,
    neutral_stiffness: float = NEUTRAL_STIFFNESS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-timestep blended stiffness, target, and damping arrays.

    When multiple gestures overlap, their parameters are blended via
    alpha-weighted averaging. When no gesture is active, neutral
    attractor parameters apply.

    Parameters
    ----------
    gestures : list[Gesture]
        List of gesture specifications.
    time : np.ndarray
        Time array in seconds.
    neutral_target : float
        Rest position when no gesture is active.
    neutral_stiffness : float
        Stiffness of the neutral attractor.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (blended_k, blended_target, blended_damping) arrays.
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
        k_sum[active] += g.stiffness * g.alpha
        target_sum[active] += g.target * g.alpha
        damping_sum[active] += g.damping * g.alpha  # type: ignore[operator]  # __post_init__ guarantees non-None

    active_mask = weight_sum > 0
    if np.any(active_mask):
        blended_k[active_mask] = k_sum[active_mask] / weight_sum[active_mask]
        blended_target[active_mask] = target_sum[active_mask] / weight_sum[active_mask]
        blended_damping[active_mask] = damping_sum[active_mask] / weight_sum[active_mask]

    return blended_k, blended_target, blended_damping


def solve_task_dynamics(
    gestures: list[Gesture],
    t_start: float = 0.0,
    t_end: float = 1.0,
    dt: float = 0.001,
    initial_position: float = 0.0,
    initial_velocity: float = 0.0,
    neutral_target: float = 0.0,
    neutral_stiffness: float = NEUTRAL_STIFFNESS,
    method: str = "LSODA",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve task dynamics for a set of gestures.

    Parameters
    ----------
    gestures : list[Gesture]
        Gestural specifications with timing and parameters.
    t_start, t_end : float
        Time range in seconds.
    dt : float
        Time step for output.
    initial_position, initial_velocity : float
        Initial state.
    neutral_target : float
        Rest position when no gesture is active.
    neutral_stiffness : float
        Stiffness of the neutral attractor.
    method : str
        ODE solver method (default LSODA).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (time, position, velocity).
    """
    time = np.arange(t_start, t_end + dt / 2, dt)
    blended_k, blended_target, blended_damping = build_blended_params(
        gestures, time, neutral_target, neutral_stiffness
    )

    def _ode(t, state):
        idx = int((t - t_start) / dt)
        idx = max(0, min(idx, len(time) - 1))
        return _sm89(t, state, blended_k[idx], blended_damping[idx], blended_target[idx])

    sol = solve_ivp(
        _ode,
        [t_start, t_end],
        [initial_position, initial_velocity],
        method=method,
        t_eval=time,
        max_step=dt,
    )
    if not sol.success:
        raise RuntimeError(f"Task dynamics solve failed: {sol.message}")

    return time, sol.y[0], sol.y[1]
