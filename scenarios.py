import numpy as np
import random
import agents
from carlo.entities import Point
import human_models


def scenario1(world, dt):
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

    # User-controlled car
    # car_human = agents.CarUserControlled(world=world, center=Point(world.p_intersection.x + 2 * world.lane_width / 4.,
    #                                                              world.p_intersection.y - 2 * world.lane_width / 2. - 3.),
    #                                    heading=np.pi / 2., color='yellow')


    # Simulated human car

    human_model = human_models.HumanModelDelayedThreshold(critical_gap=20,
                                                          noise_intensity=10,
                                                          delay_mean=0.8,
                                                          delay_std=0.3)

    human_model = human_models.HumanModelEvidenceAccumulation(critical_gap=27,
                                                              boundary=2,
                                                              drift_rate=2,
                                                              diffusion_rate=1,
                                                              dt=dt)

    car_human = agents.CarSimulatedHuman(world=world, human_model=human_model,
                                         center=Point(world.p_intersection.x + 2 * world.lane_width / 4.,
                                                      world.p_intersection.y - 2 * world.lane_width / 2. - 3.),
                                         heading=np.pi / 2., color='black', dt=dt)
    world.add(car_human)

    # add AV (hardcoded inputs)
    u = np.zeros((2, world.time_vector.shape[0]))
    u[1, 0:int(1 / dt)] = 6
    u[1, int(1 / dt):-1] = 0.05  # just a little bit of acceleration to negate friction
    car_av = agents.CarHardCoded(world=world,
                                 center=Point(world.p_intersection.x - 2 * world.lane_width / 4., world.height - 25.),
                                 heading=- np.pi / 2., control_input=u, dt=dt)
    world.add(car_av)

    # render, just to get to see our work come to life
    world.render()

    # @Arkady - my attempt, but this is suboptimal, want to use a @setter for this ideally
    # giving the cars 'world' as a paremeter doesn't seem to work for the usercontrolled car.
    # car_ego.world = world
    # car_av.world = world

    # @ Niek: hmm not sure, I added world to the constructors, seems to work for now
    # But still seems suboptimal, something to think through later

    world.run()
