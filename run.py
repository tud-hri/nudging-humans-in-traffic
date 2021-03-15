'''
Our world: an intersection with left-turn scenario
'''

import scenarios
from intersection_world import IntersectionWorld

# Run an example experiment
dt = 0.1  # 100 ms time step
end_time = 15  # run time seconds


intersection_world = IntersectionWorld(dt=dt, end_time=end_time, width=80., height=80., ppm=6., lane_width=3.)

# create a scenario
scenarios.scenario1(world=intersection_world, dt=dt)

# world = intersection_world.scenario1(dt, end_time)

# world.run()