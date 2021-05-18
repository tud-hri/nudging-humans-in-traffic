'''
Our world: an intersection with left-turn scenario
'''

import scenarios
from intersection_world import IntersectionWorld
from simulator import Simulator

if __name__ == "__main__":
    # Run an example experiment
    dt = 0.05  # 20 ms time step

    # create our world
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
    world = IntersectionWorld(dt=dt, width=60., height=120.)

    for i in range(2):
        # run a scenario in this world
        scenarios.scenario_pilot1(world=world, d0_av=60., v0_av=30. / 3.6, a_av=0.)
        sim = Simulator(world, end_time=5., ppm=8)
        sim.run()
