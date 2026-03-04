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
field.add_input("input1", amplitude=5, position=-5, width=1.0, start=50, end=150)
field.add_input("input2", amplitude=5, position=5, width=1.0, start=100, end=200)
field.solve(t_start=0, t_end=250, dt=1, tau=25.0, h=-2.0)

# plot to check activation
plot_field_heatmap(field.time, field.x, field.activation)
plot_field_surface(field.time, field.x, field.activation, threshold=0.0)

# Extract peak activation above threshold and corresponding parameter values
targets = Targets(field)
param_peak, act_peak, time = targets.peak_activation(above_threshold=True)

# solve tract variables based on peak activation value; stiffness needs to be tuned depending on duration of inputs in dynamic field; here k=8000 reaches targets
# the timescale corresponds to the period of above-threshold activation, where 0 is start of above-threshold activation. By default this will convert to milliseconds, but if you want to retain DFT scaling (i.e. dt=1.0) then k should be reduced accordingly (k=0.008). 
td = TaskDynamics()
td.solve_from_trace(time, param_peak, k=8000, time_scale=0.001)

# plot tract variable
td.plot(abs_velocity=True)
td.plot_params(params=["target"])
