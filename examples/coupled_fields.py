"""
2-layer (planning + memory) and 3-layer coupled field examples.
"""

from pyphonplan import FieldSystem
from pyphonplan.viz import plot_field_heatmap

"""
2-layer: planning + memory
"""

# initialise field system (use FieldSyste to manage multiple fields and their interactions)
sys = FieldSystem(x_min=-10, x_max=10, step_size=0.02)

# planning field with kernel params
kernel_params = dict(c_exc=25, c_inh=22.5, c_global=0.1, sigma_exc=2.0, sigma_inh=4.0, expand=3.0,)
sys.add_field("planning", tau=25, h=-2, kernel_params=kernel_params)

# memory field with kernel params
kernel_memory = dict(c_exc=5, c_inh=2.5, c_global=0.0, sigma_exc=2.0, sigma_inh=4.0, expand=3.0,)
sys.add_field("memory", tau=150, h=-5, kernel_params=kernel_memory, field_type="memory", tau_decay=500, source_field="planning")

# coupling between fields
sys.add_coupling("memory", "planning", weight=10.0)

# input to planning field
sys.add_input("planning", "response", amplitude=100, position=0, width=0.5, start=50, end=200)

# inspect field components before solving
sys.plot_kernel("planning")
sys.plot_sigmoid("planning")
sys.plot_inputs("planning")

# solve field dynamics
sys.solve(t_start=0, t_end=300)

# plot planning + memory fields
plot_field_heatmap(sys.time, sys.x, sys.activation["planning"], title="Planning field")
plot_field_heatmap(sys.time, sys.x, sys.activation["memory"], title="Memory field")


"""
3-layer: planning + memory + perception
"""

# initialise field
sys = FieldSystem(x_min=-10, x_max=10, step_size=0.05)

# planning field with kernel params
kp = dict(c_exc=25, c_inh=22.5, c_global=0.1, sigma_exc=2.0, sigma_inh=4.0)
sys.add_field("planning", tau=25, h=-2, kernel_params=kp, gamma_gated=True)

# memory field with kernel params
km = dict(c_exc=5, c_inh=2.5, c_global=0.0, sigma_exc=2.0, sigma_inh=4.0)
sys.add_field("memory", tau=150, h=-5, kernel_params=km, field_type="memory", tau_decay=500, source_field="planning")

# perception field with kernel params
kperc = dict(c_exc=15, c_inh=12, c_global=0.05, sigma_exc=2.0, sigma_inh=4.0)
sys.add_field("perception", tau=10, h=-2, kernel_params=kperc)

# add couplings
sys.add_coupling("memory", "planning", weight=10.0)
sys.add_coupling("perception", "planning", weight=100.0)  # weak: sub-threshold preshaping

# add inputs to planning and perception field with different timings
# Response cue drives planning; perception alone should preshape but not trigger
# peaks in the gamma-gated planning field (Eq. 7).
sys.add_input("perception", "auditory", amplitude=80, position=-5, width=1, start=50, end=150)
sys.add_input("planning", "response", amplitude=100, position=0, width=1, start=150, end=300)

# solve field dynamics
sys.solve(t_start=0, t_end=350)

# plot all three layers
plot_field_heatmap(sys.time, sys.x, sys.activation["planning"], title="Planning field (gamma-gated)")
plot_field_heatmap(sys.time, sys.x, sys.activation["perception"], title="Perception field")
plot_field_heatmap(sys.time, sys.x, sys.activation["memory"], title="Memory field")
