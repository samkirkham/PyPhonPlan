# PyPhonPlan

**Simulating phonetic planning with dynamic neural fields and task dynamics**

PyPhonPlan is a computational toolkit for modelling phonetic planning in terms of dynamic neural fields (Schöner et al. 2016), which then form inputs to a task dynamic equation (Saltzman & Munhall 1989) for generating tract variable trajectories. PyPhonPlan is implemented entirely in Python and is open-source software.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/samkirkham/PyPhonPlan.git
cd PyPhonPlan
uv sync
```

Then run any scripts/notebooks with `uv run`:


## Quick start

This is how you define a single-layer dynamic field, with two inputs at different time intervals. We then extract the above-threshold peak activation trace and use this as a time-varying target to drive a task dynamic equation.

```python
from pyphonplan import DynamicField, Targets, TaskDynamics
from pyphonplan.viz import plot_field_heatmap

# Single dynamic neural field with two competing inputs
field = DynamicField(x_min=-10, x_max=10, step_size=0.1)
field.set_sigmoid(beta=1.5, threshold=0.0)
field.set_kernel(c_exc=2, c_inh=1, c_global=0.5, sigma_exc=1.0, sigma_inh=2.0)
field.add_input("s1", amplitude=100, position=-5, width=1.0, start=50, end=150)
field.add_input("s2", amplitude=100, position=5, width=1.0, start=100, end=200)
field.solve(t_start=0, t_end=250, dt=1, tau=25.0, h=-2.0)

plot_field_heatmap(field.time, field.x, field.activation)

# Extract above-threshold peak trace and drive task dynamics
targets = Targets(field)
param_peak, act_peak, time = targets.peak_activation(above_threshold=True)

td = TaskDynamics()
td.solve_from_trace(time, param_peak, k=8000, time_scale=0.001)
td.plot(abs_velocity=True)
```

## Examples

The following Jupyter notebooks demonstrate the use of `PyPhonPlan` in greater detail, with a range of use cases and simulation types.

| Notebook | Description |
|---|---|
| `examples/single_field.ipynb` | Single-layer dynamic field with two competing inputs, target extraction, peak tracking, and field-to-task dynamics pipeline. |
| `examples/dual_fields.ipynb` | Two-layer coupled field system (planning + memory) with inter-field coupling. |
| `examples/tract_variables.ipynb` | Task dynamics from explicit gestural specifications, including overlapping gestures with parameter blending. |
| `examples/shadowing.ipynb` | Three-layer shadowing paradigm (baseline, shadowing, washout) demonstrating phonetic convergence due to perception-induced changes in the coupled memory field.

## Package structure

```
src/pyphonplan/
    field/
        inputs.py       GaussianInput dataclass
        kernel.py        sigmoid, interaction_kernel, convolve
        field.py         DynamicField (single field)
        coupled.py       FieldSystem (N coupled fields)
    targets/
        targets.py       Targets (target extraction + peak tracking)
    taskdynamics/
        solver.py        Gesture, solve_task_dynamics
    viz/
        field_plots.py   plot_field_heatmap, plot_field_surface, plot_inputs, plot_kernel, plot_sigmoid
        target_plots.py  plot_target_activations, plot_peak_activation
        task_plots.py    plot_trajectory, plot_blended_params
```

## Functionality

### Dynamic fields

- **`DynamicField`** — single-layer Amari field with sigmoidal thresholding, lateral inhibition kernel, Gaussian inputs with timing, and Euler-Maruyama integration with optional noise.
- **`FieldSystem`** — N coupled fields with between-field coupling. Supports standard fields and memory fields. Features include:
  - Gamma gating (`gamma_gated=True`): latches self-excitation off until direct input arrives, preventing coupled fields from triggering involuntary peaks.
  - Sigmoid or raw coupling (`sigmoid=False`): raw coupling passes activation values directly, useful when sigmoid output at resting level is too small for effective coupling.
  - Memory persistence across successive `solve()` calls for updating memory dyamics.
  - `reset()` and `clear_inputs()` for managing state between simulations.

### Target extraction

- **`Targets`** — extracts activation time series at specified spatial positions, computes onset/offset durations, and tracks peak activation over time. Peak traces can drive task dynamics.

### Task dynamics

- **`TaskDynamics`** — critically damped second-order ODE solver (Saltzman & Munhall 1989). Supports:
  - Gestural specifications with target, stiffness, timing, and blending weights.
  - Alpha-weighted parameter blending for overlapping gestures.
  - `solve_from_trace()`: drives TD directly from a field peak activation trace.
  - `add_gesture()` + `solve()`: drives TD from explicit gestural targets.

### Visualization

- Heatmaps with peak trajectory overlay and threshold crossing markers.
- 3D surface plots with threshold plane.
- Sigmoid, kernel, and input profile plots.
- Task dynamic trajectory and velocity plots.
- Field activation animations.


## Author

Sam Kirkham [s.kirkham@lancaster.ac.uk](mailto:s.kirkham@lancaster.ac.uk)