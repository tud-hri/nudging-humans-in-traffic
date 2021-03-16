import numpy as np

from agents import Car, CarUserControlled, CarSimpleMPC
from lane import HLane, VLane
from simulator import Simulator
from world import World

# a whole new woooooorld
world = World(0.1, 80., 100.)
world.lanes.append(VLane([40., 0.], [40., 120.], 3.))
world.lanes.append(VLane([37., 0.], [37., 120.], 3.))
world.lanes.append(HLane([0., 30.], [40., 30.], 3.))

# add our cars
world.agents.update({"human": CarUserControlled(p0=[40., 20.], phi0=np.pi / 2.)})
world.agents.update({"av": CarSimpleMPC(p0=[37., 90.], phi0=-np.pi / 2.)})

# create and run the simulation
sim = Simulator(world, T=20., ppm=8)
sim.run()