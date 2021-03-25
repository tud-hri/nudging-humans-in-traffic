import numpy as np

import agents
import human_models
from intersection_world import IntersectionWorld


def scenario1(world: IntersectionWorld):
    """
    Scenario 1
    Straight intersection, two approaching cars, ego vehicle turns left, automated vehicle goes straight
    :param world: world in which to run the scenario
    :param dt: time step
    """

    # add ego vehicle (hardcoded inputs)
    # at stop sign at the intersection, accelerated and turns left
    # u = np.zeros((2, world.time_vector.shape[0]))
    # idx_turn = random.randint(0 / dt, 3 / dt)
    # u[0, idx_turn:idx_turn + int(1.45 / dt)] = 0.455
    # u[1, idx_turn:idx_turn + int(4 / dt)] = 0.5
    # car_human = agents.CarHardCoded(center=Point(world.p_intersection.x + 2 * world.lane_width / 4.,
    #                                            world.p_intersection.y - 2 * world.lane_width / 2. - 3.),
    #                               heading=np.pi / 2., control_input=u, color='blue')

    # simulated human car
    human_model = human_models.HumanModelDelayedThreshold(critical_gap=20,
                                                          noise_intensity=10,
                                                          delay_mean=0.8,
                                                          delay_std=0.3)

    human_model = human_models.HumanModelEvidenceAccumulation(critical_gap=27,
                                                              boundary=2,
                                                              drift_rate=2,
                                                              diffusion_rate=1,
                                                              dt=world.dt)

    car_human = agents.CarSimulatedHuman(p0=[40., 20.], phi0=np.pi / 2., world=world, human_model=human_model, dt=world.dt, color='red')
    world.agents.update({'human': car_human})

    # add AV (hardcoded inputs)
    # u = np.zeros((2, world.time_vector.shape[0]))
    # u[1, 0:int(1 / world.dt)] = 6
    # u[1, int(1 / world.dt):-1] = 0.05  # just a little bit of acceleration to compensate for friction
    car_av = agents.CarMPC(p0=[37., 65.], phi0=-np.pi / 2., v0=30 / 3.6, world=world, dt=world.dt, color='yellow')
    world.agents.update({'av': car_av})
