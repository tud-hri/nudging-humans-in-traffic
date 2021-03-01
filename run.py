'''
Our world: an intersection with left-turn scenario
'''

import world
import time

# Run an example experiment
dt = 0.1  # 100 ms time step
end_time = 15  # run time seconds

# create a scenario
world = world.scenario1(dt, end_time)

world.run()