"""
Task dynamics example
"""

from pyphonplan import TaskDynamics

# initialise simulation object with start/end times and time-step
# if end time exceeds the final gesture end time then it will return to a neutral value of 0, to avoid this make sure that t_end matches the final gesture end time
td = TaskDynamics(t_start=0.0, t_end=0.5, dt=0.001)

# add two gestures with different targets, stiffnesses, timings and alphas
# note that an alpha ratio of 1:100 means that gesture 2 will dominate during blending
td.add_gesture(target=5.0, k=2000.0, alpha=1, start=0.05, end=0.25)
td.add_gesture(target=-5.0, k=2000.0, alpha=100, start=0.15, end=0.40)
td.solve()

# plot tract variable and velocity
td.plot(abs_velocity=True)

# plot (blended) parameter values over time
# default is params=["target", "k", "damping"]
td.plot_params()
td.plot_params(params=["target", "k"])
