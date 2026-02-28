"""
Basic single-field dynamic neural field simulation
"""

from pyphonplan import DynamicField, Targets
from pyphonplan.viz import plot_field_heatmap, plot_target_activations, plot_field_surface, animate_field

# Create field
field = DynamicField(x_min=-10, x_max=10, step_size=0.1)
field.set_sigmoid(beta=1.5, threshold=0.0)
field.set_kernel(c_exc=1, c_inh=1, c_global=0.05, sigma_exc=1.0, sigma_inh=5.0, expand=3.0)

# visualise sigmoid and kernel
field.plot_sigmoid()
field.plot_kernel()

# Add two competing inputs with different timing
field.add_input("input1", amplitude=5, position=-5, width=1.0, start=50, end=150)
field.add_input("input2", amplitude=5, position=5, width=1.0, start=100, end=200)
field.plot_inputs()

# Solve
field.solve(t_start=0, t_end=250, dt=1, tau=25.0, h=-2.0, noise=1.0)

# plot field activation over time and space
plot_field_heatmap(field.time, field.x, field.activation)

# plot field activation surface                                                          
plot_field_surface(field.time, field.x, field.activation, threshold=0.0)   

# Extract targets at input positions
targets = Targets(field, positions=[-5.0, 5.0])
plot_target_activations(field.time, targets.traces)
targets.peak_activation(plot=True)

# Animate field activation over time (save as: save_path="single_field.mp4")
animate_field(field.time, field.x, field.activation)