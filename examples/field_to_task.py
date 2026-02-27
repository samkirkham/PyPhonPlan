"""
Field-to-task dynamics pipeline

Demonstrates driving a task dynamic solver from a dynamic field's
peak activation trace, connecting the DynamicField → Targets →
TaskDynamics pipeline end-to-end.
"""

from pyphonplan import DynamicField, Targets, TaskDynamics

# Dynamic field simulation
field = DynamicField(x_min=-10, x_max=10, step_size=0.1)
field.set_sigmoid(beta=1.5, threshold=0.0)
field.set_kernel(c_exc=1, c_inh=1, c_global=0.05, sigma_exc=1.0, sigma_inh=5.0, expand=3.0)
field.add_input("inp", amplitude=5, position=5, width=1.0, start=50, end=150)
field.solve(t_start=0, t_end=250, dt=1, tau=25.0, h=-2.0)

# Extract target trace
targets = Targets(field, positions=[5.0])

# above_threshold=False gives a regularly-spaced time array needed by the ODE
# NOTE: this is not ideal -> need to use above_threshold values but also handle edge cases in which there's a gap in above threshold activation at some point -> about to refactor this.
param_peak, act_peak, time = targets.peak_activation(above_threshold=False)

td = TaskDynamics()
td.solve_from_trace(time, param_peak, k=2000.0)

td.plot()
td.plot_params(params=["target"])
