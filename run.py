'''
Our world: an intersection with left-turn scenario
'''

import scenarios
from intersection_world import IntersectionWorld

# Run an example experiment
dt = 0.02  # 20 ms time step
end_time = 10  # run time seconds

# create our world
# coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
intersection_world = IntersectionWorld(dt=dt, end_time=end_time, width=80., height=80., ppm=6., lane_width=3.)

# run a scenario in this world
scenarios.scenario1(world=intersection_world, dt=dt)
