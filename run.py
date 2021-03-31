'''
Our world: an intersection with left-turn scenario
'''

import scenarios
from intersection_world import IntersectionWorld
from simulator import Simulator

# Run an example experiment
dt = 0.1  # 20 ms time step

# create our world
# coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
world = IntersectionWorld(dt=dt, width=80., height=80.)

for i in range(5):
    # run a scenario in this world
    scenarios.scenario1(world=world)
    sim = Simulator(world, T=7., dt=dt, ppm=8)
    sim.run()
