"""Coupled multi-layer dynamic field system.

Supports N named fields with inter-field coupling. Field types:
- "standard": standard Amari dynamics
- "memory": conditional build/decay driven by sigmoid of a source field

Based on dft_memory and dft_perception in
phonology_wo_symbols/figures/dnf/functions.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from pyphonplan.field.inputs import GaussianInput
from pyphonplan.field.kernel import interaction_kernel, sigmoid, convolve, make_kernel_x


@dataclass
class _FieldSpec:
    """Internal specification for a single field in the system."""

    name: str
    tau: float
    h: float
    kernel: np.ndarray
    field_type: str = "standard"
    sigmoid_beta: float = 1.5
    sigmoid_threshold: float = 0.0
    tau_decay: float | None = None
    source_field: str | None = None
    gamma_gated: bool = False
    inputs: dict[str, GaussianInput] = dc_field(default_factory=dict)


@dataclass
class _Coupling:
    """Directed coupling between two fields."""

    source: str
    target: str
    weight: float
    use_sigmoid: bool = True


class FieldSystem:
    """N coupled named dynamic fields.

    Each field has a name, type, and independent parameters. Fields are
    coupled via add_coupling(source, target, weight).

    Parameters
    ----------
    x_min, x_max : float
        Shared spatial range for all fields.
    step_size : float
        Spatial resolution.
    """

    def __init__(self, x_min: float = -10.0, x_max: float = 10.0, step_size: float = 0.1):
        self.x = np.arange(x_min, x_max + step_size / 2, step_size)
        self._dx = step_size
        self._fields: dict[str, _FieldSpec] = {}
        self._field_order: list[str] = []
        self._couplings: list[_Coupling] = []
        self._last_state: dict[str, np.ndarray] = {}
        self.time: np.ndarray | None = None
        self.activation: dict[str, np.ndarray] = {}

    @property
    def n_x(self) -> int:
        return len(self.x)

    def add_field(
        self,
        name: str,
        *,
        tau: float,
        h: float,
        kernel_params: dict,
        field_type: str = "standard",
        sigmoid_beta: float = 1.5,
        sigmoid_threshold: float = 0.0,
        tau_decay: float | None = None,
        source_field: str | None = None,
        gamma_gated: bool = False,
    ):
        """Add a named field to the system.

        Parameters
        ----------
        name : str
            Unique identifier for this field.
        tau : float
            Time constant.
        h : float
            Resting level.
        kernel_params : dict
            Passed to interaction_kernel: c_exc, c_inh, c_global, sigma_exc, sigma_inh,
            and optionally mu, expand.
        field_type : str
            "standard" or "memory".
        sigmoid_beta, sigmoid_threshold : float
            Sigmoid parameters for this field.
        tau_decay : float or None
            Decay time constant (memory fields only).
        source_field : str or None
            Name of the field whose sigmoid drives memory update.
        gamma_gated : bool
            If True, the self-excitation kernel is latched off until direct input
            arrives (s > 0). The latch stays open while activation exceeds
            threshold, and activation is clamped at threshold when the latch
            is closed.
        """
        if field_type == "memory":
            if source_field is None:
                raise ValueError("Memory fields require source_field.")
            if tau_decay is None:
                raise ValueError("Memory fields require tau_decay.")

        expand = kernel_params.get("expand", 3.0)
        mu = kernel_params.get("mu", 0.0)
        kernel_x = make_kernel_x(self.x, expand)
        kernel = interaction_kernel(
            kernel_x,
            kernel_params["c_exc"],
            kernel_params["c_inh"],
            kernel_params["c_global"],
            kernel_params["sigma_exc"],
            kernel_params["sigma_inh"],
            mu,
        )

        spec = _FieldSpec(
            name=name,
            tau=tau,
            h=h,
            kernel=kernel,
            field_type=field_type,
            sigmoid_beta=sigmoid_beta,
            sigmoid_threshold=sigmoid_threshold,
            tau_decay=tau_decay,
            source_field=source_field,
            gamma_gated=gamma_gated,
        )
        self._fields[name] = spec
        self._field_order.append(name)

    def plot_sigmoid(self, field_name: str, show: bool = True):
        """Plot the sigmoid thresholding function for a named field."""
        from pyphonplan.viz.field_plots import plot_sigmoid
        spec = self._fields[field_name]
        return plot_sigmoid(self.x, spec.sigmoid_beta, spec.sigmoid_threshold, show=show)

    def plot_kernel(self, field_name: str, show: bool = True):
        """Plot the interaction kernel for a named field."""
        from pyphonplan.viz.field_plots import plot_kernel
        spec = self._fields[field_name]
        return plot_kernel(self.x, spec.kernel, show=show)

    def plot_inputs(self, field_name: str, show: bool = True):
        """Plot all Gaussian input profiles for a named field."""
        spec = self._fields[field_name]
        if not spec.inputs:
            raise RuntimeError(f"No inputs added to field '{field_name}'.")
        from pyphonplan.viz.field_plots import plot_inputs
        return plot_inputs(self.x, spec.inputs, show=show)

    def add_coupling(self, source: str, target: str, weight: float, sigmoid: bool = True):
        """Add directed coupling from source to target.

        Parameters
        ----------
        source, target : str
            Field names.
        weight : float
            Coupling strength.
        sigmoid : bool
            If True (default), couple via sigmoid(u_source). If False,
            couple via raw u_source. Memory sources always use raw coupling
            regardless of this setting.
        """
        self._couplings.append(_Coupling(source, target, weight, use_sigmoid=sigmoid))

    def add_input(
        self,
        field_name: str,
        input_name: str,
        *,
        amplitude: float,
        position: float,
        width: float,
        offset: float = 0.0,
        start: int = 0,
        end: int = 0,
    ):
        """Add a Gaussian input to a specific field."""
        inp = GaussianInput(
            name=input_name,
            amplitude=amplitude,
            position=position,
            width=width,
            offset=offset,
            start=start,
            end=end,
        )
        self._fields[field_name].inputs[input_name] = inp

    def _sum_inputs_at(self, field_name: str, t: int) -> np.ndarray:
        """Sum active inputs for a field at time step t."""
        total = np.zeros(self.n_x)
        for inp in self._fields[field_name].inputs.values():
            if inp.is_active(t):
                total += inp.evaluate(self.x)
        return total

    def _get_couplings_for(self, target: str, state_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Sum coupling contributions to target field.

        Memory source fields always couple raw (weight * u_mem). Other
        source fields couple via sigmoid(u) by default, or raw if the
        coupling was created with sigmoid=False.
        """
        total = np.zeros(self.n_x)
        for c in self._couplings:
            if c.target == target:
                source_spec = self._fields[c.source]
                if source_spec.field_type == "memory" or not c.use_sigmoid:
                    total += c.weight * state_dict[c.source]
                else:
                    gu = sigmoid(state_dict[c.source],
                                 source_spec.sigmoid_beta,
                                 source_spec.sigmoid_threshold)
                    total += c.weight * gu
        return total

    def solve(
        self,
        t_start: int,
        t_end: int,
        dt: int = 1,
        y0: dict[str, np.ndarray] | None = None,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        """Solve the coupled field system using Euler-Maruyama integration.

        After solving, `self.time` and `self.activation[field_name]` are set.

        Parameters
        ----------
        t_start, t_end : int
            Time range (integer time steps).
        dt : int
            Time step.
        y0 : dict mapping field name to initial state, or None.
            If None, each field starts at its resting level h.
        noise : float
            Noise amplitude.
        rng : np.random.Generator or None
            Random number generator for reproducibility. If None, uses
            ``np.random.default_rng()``.
        """
        n = self.n_x

        self.time = np.arange(t_start, t_end + dt, dt)
        n_steps = len(self.time)

        # Initialise state and storage per field
        results = {name: np.empty((n, n_steps)) for name in self._field_order}
        state = {}
        for name in self._field_order:
            if y0 is not None and name in y0:
                state[name] = y0[name].copy()
            elif name in self._last_state:
                state[name] = self._last_state[name].copy()
            else:
                if self._fields[name].field_type == "memory":
                    state[name] = np.zeros(n)
                else:
                    state[name] = self._fields[name].h * np.ones(n)
            results[name][:, 0] = state[name]

        noise_scale = noise * np.sqrt(dt) if noise > 0 else 0.0
        if noise_scale > 0 and rng is None:
            rng = np.random.default_rng()
        gate_latched = {name: False for name in self._field_order}

        for i in range(1, n_steps):
            t = self.time[i - 1]
            nearest_t = round(t)

            for name in self._field_order:
                spec = self._fields[name]
                u = state[name]

                if spec.field_type == "standard":
                    s = self._sum_inputs_at(name, nearest_t)
                    coupling = self._get_couplings_for(name, state)
                    gu = sigmoid(u, spec.sigmoid_beta, spec.sigmoid_threshold)
                    kernel_term = self._dx * convolve(gu, spec.kernel)
                    if spec.gamma_gated:
                        if np.any(s > 0):
                            gate_latched[name] = True
                        elif not np.any(u > spec.sigmoid_threshold):
                            gate_latched[name] = False
                    dudt = (-u + spec.h + s + coupling + kernel_term) / spec.tau
                    new_u = u + dt * dudt
                    if noise_scale > 0:
                        new_u += (noise_scale / spec.tau) * rng.normal(0, 1, n)
                    if spec.gamma_gated and not gate_latched[name]:
                        new_u = np.minimum(new_u, spec.sigmoid_threshold)

                elif spec.field_type == "memory":
                    source_u = state[spec.source_field]
                    source_spec = self._fields[spec.source_field]
                    gu_source = sigmoid(source_u, source_spec.sigmoid_beta, source_spec.sigmoid_threshold)
                    dudt = np.where(
                        source_u > source_spec.sigmoid_threshold,
                        (-u + self._dx * convolve(gu_source, spec.kernel)) / spec.tau,
                        -u / spec.tau_decay,
                    )
                    new_u = u + dt * dudt

                else:
                    raise ValueError(f"Unknown field_type: {spec.field_type!r}")

                state[name] = new_u
                results[name][:, i] = new_u

        # Persist final state for memory fields
        for name in self._field_order:
            if self._fields[name].field_type == "memory":
                self._last_state[name] = state[name].copy()

        self.activation = results

    def reset(self, field_name: str | None = None):
        """Clear stored state so next solve() starts from resting level.

        If field_name is given, reset only that field. Otherwise reset all.
        """
        if field_name is not None:
            self._last_state.pop(field_name, None)
        else:
            self._last_state.clear()

    def clear_inputs(self, field_name: str | None = None):
        """Remove all inputs from a field, or all fields if no name given."""
        if field_name is not None:
            self._fields[field_name].inputs.clear()
        else:
            for spec in self._fields.values():
                spec.inputs.clear()
