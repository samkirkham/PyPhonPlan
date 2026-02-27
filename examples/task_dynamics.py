"""
Task dynamics example
"""

from pyphonplan import Gesture, solve_task_dynamics
from pyphonplan.task.solver import build_blended_params
from pyphonplan.viz import plot_trajectory, plot_blended_params

# Two overlapping gestures
gestures = [
    Gesture(target=5.0, stiffness=100.0, start=0.05, end=0.25),
    Gesture(target=-3.0, stiffness=80.0, start=0.15, end=0.40),
]

time, position, velocity = solve_task_dynamics(
    gestures,
    t_start=0.0,
    t_end=0.5,
    dt=0.001,
    initial_position=0.0,
    neutral_target=0.0,
)


# Build blended params
blended_k, blended_target, blended_damping = build_blended_params(
    gestures, time, neutral_target=0.0
)

# Plots
plot_trajectory(time, position, velocity)
plot_blended_params(time, blended_k, blended_target, blended_damping)
