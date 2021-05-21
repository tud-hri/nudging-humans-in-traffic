import numpy as np

import agents
import human_models
from intersection_world import IntersectionWorld


def scenario_demo_1(world: IntersectionWorld):
    """
    Demo scenario
    Straight intersection, two approaching cars, ego vehicle controlled by a simulated human turns left,
    automated vehicle goes straight
    :param world: world in which to run the scenario
    """

    # simulated human car
    # human_model = human_models.HumanModelDelayedThreshold(critical_gap=20,
    #                                                       noise_intensity=10,
    #                                                       delay_mean=0.8,
    #                                                       delay_std=0.3)

    human_model = human_models.HumanModelDDMStaticDrift(critical_gap=40,
                                                        boundary=1,
                                                        drift_rate=0.5,
                                                        diffusion_rate=1,
                                                        dt=world.dt)

    car_human = agents.CarSimulatedHuman(p0=[40., 20.], v0=0., phi0=np.pi / 2., world=world, human_model=human_model,
                                         color='red')
    world.agents.update({'human': car_human})

    # add AV
    car_av = agents.CarMPC(p0=[37., 65.], phi0=-np.pi / 2., v0=25 / 3.6, world=world, color='blue')
    world.agents.update({'av': car_av})

    # specify the
    theta_human = [1., 1., -5., -3., 500., 2000., 1.]  # velocity, heading, primary_lane, all_lanes, road_shoulder, obstacle, input
    car_human.set_objective(theta=theta_human, primary_lanes=[world.lanes[0], world.lanes[2]],
                            all_lanes=world.lanes, obstacles=[car_av], road_shoulders=[world.shoulders[0], world.shoulders[2]],
                            heading=np.pi)

    theta_av = [0.25, 2., -5., -3., 500., 2000., 1.]  # velocity, heading, primary_lane, all_lanes, road_shoulder, obstacle, input
    car_av.set_objective(theta=theta_av, primary_lanes=[world.lanes[3]], all_lanes=[world.lanes[2], world.lanes[3]], obstacles=[car_human],
                         road_shoulders=[world.shoulders[1], world.shoulders[2]], heading=-np.pi / 2.)

def scenario_pilot1(world: IntersectionWorld, d0_av, v0_av, a_av):
    """
    scenario for pilot 1
    :param world:
    :return:
    """

    y_crossing = 28.5  # y position of the crossing (hardcoded for now, ideally we should get it from the intersection world)

    # Add AV
    # create an AV agent, set its initial position and velocity, then set its constant input (acceleration)
    car_av = agents.Car(p0=[37., y_crossing + d0_av], phi0=-np.pi / 2., v0=v0_av, world=world)
    u_av = np.zeros((3, 1))  # constant input [accelerate (m/s2), decelerate (m/s2), steering (rad)] for the AV
    if np.sign(a_av) == 1:
        u_av[0] = a_av
    elif np.sign(a_av) == -1:
        u_av[1] = a_av
    car_av.u = u_av
    world.agents.update({'av': car_av})

    # add human
    car_human = agents.CarHumanInitiatedPD(p0=[40., 23.], v0=0., phi0=np.pi / 2., world=world, color='blue')
    world.agents.update({'human': car_human})

