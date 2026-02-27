"""Gaussian input specification for dynamic fields."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianInput:
    """A time-gated Gaussian input to a dynamic field.

    Parameters
    ----------
    name : str
        Identifier for this input.
    amplitude : float
        Peak amplitude of the Gaussian.
    position : float
        Centre position along the field dimension.
    width : float
        Standard deviation of the Gaussian.
    offset : float
        Constant offset added to the Gaussian profile.
    start : int
        Time step at which input becomes active (inclusive).
    end : int
        Time step at which input becomes inactive (inclusive).
    """

    name: str
    amplitude: float
    position: float
    width: float
    offset: float = 0.0
    start: int = 0
    end: int = 0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Compute Gaussian profile over spatial array x.

        Returns a * exp(-(x - p)^2 / (2 * w^2)) + offset.
        """
        return self.amplitude * np.exp(
            -((x - self.position) ** 2) / (2 * self.width**2)
        ) + self.offset

    def is_active(self, t: int) -> bool:
        """True if the input is active at time step t."""
        return self.start <= t <= self.end
