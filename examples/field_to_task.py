"""
Field-to-task dynamics pipeline

Demonstrates driving a task dynamic solver from a dynamic field's
peak activation trace, connecting the DynamicField → Targets →
TaskDynamics pipeline end-to-end.
"""

from pyphonplan import DynamicField, Targets, TaskDynamics
from pyphonplan.viz import plot_field_heatmap, plot_field_surface

# Dynamic field simulation
field = DynamicField(x_min=-10, x_max=10, step_size=0.1)
field.set_sigmoid(beta=1.5, threshold=0.0)
field.set_kernel(c_exc=1, c_inh=1, c_global=0.05, sigma_exc=1.0, sigma_inh=5.0, expand=3.0)
field.add_input("inp1", amplitude=5, position=5, width=1.0, start=50, end=150)
field.add_input("inp2", amplitude=5, position=-5, width=1.0, start=100, end=200)
field.solve(t_start=0, t_end=250, dt=1, tau=25.0, h=-2.0)

# plot to check activation
plot_field_heatmap(field.time, field.x, field.activation)
plot_field_surface(field.time, field.x, field.activation, threshold=0.0)

# Extract target trace (above_threshold filters to active period only)
targets = Targets(field)
param_peak, act_peak, time = targets.peak_activation(above_threshold=True)

td = TaskDynamics()
td.solve_from_trace(time, param_peak, k=2000.0)

# plot tract variable
# note that timescale is now in seconds, rather than ms (as in the field). This is largely due to conventions in each field, but 
td.plot(abs_velocity=True)
td.plot_params(params=["target"])
