'''
Our world: an intersection with left-turn scenario
'''

import numpy as np

import scenarios
from intersection_world import IntersectionWorld
from simulator import Simulator

if __name__ == "__main__":
    # Run an example experiment
    dt = 0.02  # 20 ms time step

    # create our world
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
    world = IntersectionWorld(dt=dt, width=60., height=120.)

    # create experiment conditions (just some numbers)
    accs = [-0.2, 0., 0.2]
    v0s = [60. / 3.6, 80. / 3.6, 100. / 3.6]
    d0s = [60., 80., 100.]

    n_repetitions = 10  # this needs to be implemented still

    # generate all possible combinations of accs, v0s, d0s
    combs = np.array(np.meshgrid(d0s, v0s, accs)).T.reshape(-1, 3)
    # randomize order
    order = np.random.permutation(combs.shape[0])

    conditions = combs[order, :]

    for i in range(0, conditions.shape[0]):
        # set conditions
        d0_av = conditions[i, 0]
        v0_av = conditions[i, 1]
        a_av = conditions[i, 2]

        # run a scenario in this world
        scenarios.scenario_pilot1(world=world, d0_av=d0_av, v0_av=v0_av, a_av=a_av)
        sim = Simulator(world, end_time=10., ppm=8)

        # run stuff
        sim.run()
        # and save stuff (just a proposal for filename coding)
        sim.save_stuff(filename="d{0:d}_v{1:d}_a{2:d}".format(int(d0_av), int(v0_av), int(a_av)))
