import numpy as np
import random
import agents
from carlo.entities import Point

def scenario1(world, dt):
    """
    Scenario 1
    Straight intersection, two approaching cars, ego vehicle turns left, automated vehicle goes straight
    :param dt: time step
    :param end_time: end time of simulation
    """

    # create our world
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
    # world = IntersectionWorld(dt, end_time, width=80., height=80., ppm=6., lane_width=3., p_intersection=Point(50., 30.))  # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    # world.create_intersection(p_intersection=p_intersection, lane_width=lane_width)  # create the intersection

    # cars
    # add ego vehicle (hardcoded inputs)
    # at stop sign at the intersection, accelerated and turns left
    u = np.zeros((2, world.time_vector.shape[0]))
    idx = random.randint(5, 10)
    u[0, idx + int(5 / dt):idx + int(11 / dt)] = 0.455
    u[1, idx + int(3 / dt):idx + int(12 / dt)] = 0.5
    # car_ego = CarHardCoded(center=Point(p_intersection.x + 2 * lane_width / 4., p_intersection.y - 2 * lane_width / 2. - 3.), heading=np.pi / 2., world=world,
    #                        input=u, color='blue')
    car_ego = agents.CarUserControlled(center=Point(world.p_intersection.x + 2 * world.lane_width / 4.,
                                                    world.p_intersection.y - 2 * world.lane_width / 2. - 3.),
                                       heading=np.pi / 2., color='yellow')
    world.add(car_ego)

    # add AV (hardcoded inputs)
    u = np.zeros((2, world.time_vector.shape[0]))
    u[1, 0:int(10 / dt)] = 0.5
    u[1, int(10 / dt):-1] = 0.05  # just a little bit of acceleration to negate friction
    car_av = agents.CarHardCoded(center=Point(world.p_intersection.x - 2 * world.lane_width / 4., world.height - 25.),
                                 heading=- np.pi / 2., input=u)
    world.add(car_av)

    # render, just to get to see our work come to life
    world.render()

    # @Arkady - my attempt, but this is suboptimal, want to use a @setter for this ideally
    # giving the cars 'world' as a paremeter doesn't seem to work for the usercontrolled car.
    car_ego.world = world
    car_av.world = world

    world.run()

    # return world