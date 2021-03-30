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

    car_human = agents.CarSimulatedHuman(p0=[40., 20.], v0=0., phi0=np.pi / 2., world=world, human_model=human_model, dt=world.dt, color='red')
    world.agents.update({'human': car_human})

    # add AV
    car_av = agents.CarMPC(p0=[40., 65.], phi0=-np.pi / 2., v0=10 / 3.6, world=world, dt=world.dt, color='yellow')
    world.agents.update({'av': car_av})

    # add rewards to cars
    theta_human = [2., 1., -5., -3., 100., 0., 1.]  # velocity, heading, primary_lane, all_lanes, road_shoulder, obstacle, input
    car_human.set_objective(theta=theta_human, primary_lanes=[world.lanes[0], world.lanes[2]], all_lanes=world.lanes, road_shoulders=[world.shoulders[0], world.shoulders[2]],
                            heading=np.pi)  #

    theta_av = [1., 2., -5., -4., 100., 1000., 1.]  # velocity, heading, primary_lane, all_lanes, road_shoulder, obstacle, input
    car_av.set_objective(theta=theta_av, primary_lanes=[world.lanes[3]], all_lanes=[world.lanes[2], world.lanes[3]], obstacles=[car_human],
                         road_shoulders=[world.shoulders[1], world.shoulders[2]], heading=-np.pi / 2.)
