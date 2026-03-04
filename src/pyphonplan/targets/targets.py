"""Target extraction and blending from solved dynamic fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetTrace:
    """Activation trace at a single target position.

    Attributes
    ----------
    position : float
        Target position along the field dimension.
    activation : np.ndarray
        1D time series of activation at this position.
    onset_idx : int or None
        Index where activation first crosses threshold.
    offset_idx : int or None
        Index where activation returns below threshold.
        None if activation never drops back below threshold.
    onset_time : float or None
        Time value at onset.
    offset_time : float or None
        Time value at offset. None if activation is still active
        at the end of the simulation.
    duration : float or None
        offset_time - onset_time. None if offset was not detected.
    """

    position: float
    activation: np.ndarray
    onset_idx: int | None = None
    offset_idx: int | None = None
    onset_time: float | None = None
    offset_time: float | None = None
    duration: float | None = None


def _find_closest_idx(array: np.ndarray, value: float) -> int:
    """Index of closest value in array."""
    return int(np.argmin(np.abs(array - value)))


class Targets:
    """Extract and analyse N target positions from a solved field.

    Parameters
    ----------
    field : object
        A solved DynamicField (or FieldSystem activation array).
        Must have attributes: x, time, activation (n_x x n_time).
    positions : list[float]
        Target positions to track.
    activation : np.ndarray or None
        If provided, use this instead of field.activation (useful for
        FieldSystem where activation is accessed by name).
    """

    def __init__(
        self,
        field,
        positions: list[float] | None = None,
        activation: np.ndarray | None = None,
    ):
        self.x = field.x
        self.time = field.time
        self.activation = activation if activation is not None else field.activation
        self.positions = positions if positions is not None else []
        self.traces: list[TargetTrace] = []

        for pos in self.positions:
            idx = _find_closest_idx(self.x, pos)
            trace = TargetTrace(
                position=pos,
                activation=self.activation[idx],
            )
            self.traces.append(trace)

    def calculate_durations(self, threshold: float = 0.0, pad: int = 10):
        """Find onset/offset for each target based on threshold crossing.

        Parameters
        ----------
        threshold : float
            Activation threshold for onset/offset detection.
        pad : int
            Minimum samples after onset before looking for offset.
        """
        for trace in self.traces:
            a = trace.activation
            above = a >= threshold

            if not np.any(above):
                continue

            crossings = np.diff(above.astype(int)) > 0  # 0→1 transitions only
            if not np.any(crossings):
                continue
            onset_idx = int(np.argmax(crossings)) + 1
            trace.onset_idx = onset_idx
            trace.onset_time = float(self.time[onset_idx])

            # Find offset: first point below threshold after onset + pad
            search_start = onset_idx + pad
            if search_start < len(a):
                remaining = a[search_start:]
                below = remaining < threshold
                if np.any(below):
                    offset_idx = search_start + int(np.argmax(below))
                    trace.offset_idx = offset_idx
                    trace.offset_time = float(self.time[offset_idx])
                    trace.duration = trace.offset_time - trace.onset_time
                # else: activation never drops — leave offset/duration as None

    def peak_activation(
        self, above_threshold: bool = True, plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get peak activation across the full field per time step.

        Operates on the full field activation, not just the target positions.

        Parameters
        ----------
        above_threshold : bool
            If True, only return time steps where peak activation >= 0.
        plot : bool
            If True, plot peak position and activation over time.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            (parameter_peak, activation_peak, time). The x-position of
            peak activation, the peak activation value, and corresponding time.
        """
        peak_indices = np.argmax(self.activation, axis=0)
        parameter_peak = self.x[peak_indices]
        activation_peak = self.activation[peak_indices, np.arange(self.activation.shape[1])]

        if above_threshold:
            mask = activation_peak >= 0
            parameter_peak = parameter_peak[mask]
            activation_peak = activation_peak[mask]
            time = self.time[mask]
        else:
            time = self.time

        if plot:
            from pyphonplan.viz.target_plots import plot_peak_activation
            plot_peak_activation(parameter_peak, activation_peak, time,
                                 dimension="parameter")
            plot_peak_activation(parameter_peak, activation_peak, time,
                                 dimension="activation")

        return parameter_peak, activation_peak, time
