'''
Our world: an intersection with left-turn scenario
'''

import random
import time

import numpy as np

from carlo.agents import Car, Painting, RectangleBuilding
from carlo.geometry import Point
from carlo.world import World


def create_scenario(time_step, world_width=80, world_height=80, pos_intersection=Point(40, 40)):
    '''
    Create the left-turn scenario
    :param time_step: in seconds
    :param pos_intersection: position of the intersection
    :return: world and dictionary of actors
    '''

    # create our worlds
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0, rad)
    world = World(dt, width=world_width, height=world_height, ppm=6)  # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # create the intersection
    n_lanes = 2
    lane_width = 3  # meters
    road_width = n_lanes * lane_width
    sidewalk_width = 2

    # in carlo, roads are defined by the sidewalks, so, create sidewalks, buildings
    # this is a bit of a mess
    # bottom left
    world.add(Painting(pos_intersection / 2 - Point(road_width, road_width) / 2, pos_intersection, 'gray80'))
    world.add(RectangleBuilding(pos_intersection / 2 - Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2, pos_intersection, 'forest green'))

    # bottom right
    world.add(Painting(Point(pos_intersection.x + (world_width - pos_intersection.x) / 2 + road_width / 2, pos_intersection.y / 2 - road_width / 2),
                       Point(world_width - pos_intersection.x, pos_intersection.y), 'gray80'))  # sidewalk
    world.add(RectangleBuilding(Point(pos_intersection.x + (world_width - pos_intersection.x) / 2 + (road_width + sidewalk_width) / 2,
                                      pos_intersection.y / 2 - (road_width + sidewalk_width) / 2),
                                Point(world_width - pos_intersection.x, pos_intersection.y), 'forest green'))  # building / grass

    # top left
    world.add(Painting(pos_intersection / 2 + Point(0, world_height / 2) + Point(-road_width, road_width) / 2,
                       Point(pos_intersection.x, world_height - pos_intersection.y), 'gray80'))
    world.add(Painting(pos_intersection / 2 + Point(0, world_height / 2) + Point(-(road_width + sidewalk_width), (road_width + sidewalk_width)) / 2,
                       Point(pos_intersection.x, world_height - pos_intersection.y), 'forest green'))

    # top right
    pos_topright = Point(world_width, world_height)
    world.add(Painting(pos_topright - (pos_topright - pos_intersection) / 2 + Point(road_width, road_width) / 2, pos_topright - pos_intersection, 'gray80'))
    world.add(RectangleBuilding(pos_topright - (pos_topright - pos_intersection) / 2 + Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2,
                                pos_topright - pos_intersection, 'forest green'))

    # adding lane markings to make things pretty
    lane_marker_length = 2
    lane_marker_width = 0.2

    # north-south road
    for y in np.arange(0, world_height + 2 * lane_marker_length, 2 * lane_marker_length):
        world.add(Painting(Point(pos_intersection.x, y), Point(lane_marker_length, lane_marker_width), 'white', heading=np.pi / 2))

    # east-west road
    for x in np.arange(0, world_width + 2 * lane_marker_length, 2 * lane_marker_length):
        world.add(Painting(Point(x, pos_intersection.y), Point(lane_marker_length, lane_marker_width), 'white', heading=- np.pi))

    # small patch to cover the lane markings at the intersection
    world.add(Painting(pos_intersection, Point(road_width, road_width), 'gray'))

    # stop lines
    world.add(Painting(pos_intersection + Point(road_width / 4, -road_width / 2 - 0.15), Point(road_width / 2, 0.3), 'white'))
    world.add(Painting(pos_intersection + Point(0, -1 - road_width / 2), Point(0.3, 2), 'white'))
    world.add(Painting(pos_intersection + Point(-road_width / 4, road_width / 2 + 0.15), Point(road_width / 2, 0.3), 'white'))
    world.add(Painting(pos_intersection + Point(0, 1 + road_width / 2), Point(0.3, 2), 'white'))
    world.add(Painting(pos_intersection + Point(road_width / 2 + 0.15, road_width / 4), Point(road_width / 2, 0.3), 'white', heading=np.pi / 2))
    world.add(Painting(pos_intersection + Point(1 + road_width / 2, 0), Point(0.3, 2), 'white', heading=np.pi / 2))
    world.add(Painting(pos_intersection + Point(-road_width / 2 - 0.15, -road_width / 4), Point(road_width / 2, 0.3), 'white', heading=np.pi / 2))
    world.add(Painting(pos_intersection + Point(-1 - road_width / 2, 0), Point(0.3, 2), 'white', heading=np.pi / 2))

    # actors
    actors = {}

    # add ego vehicle (at stop sign at the intersection)
    actors['ego'] = Car(Point(pos_intersection.x + road_width / 4, pos_intersection.y - road_width / 2 - 3), np.pi / 2, 'blue')
    world.add(actors['ego'])

    # add AV
    actors['av'] = Car(Point(pos_intersection.x - road_width / 4, world_height - 25), - np.pi / 2)
    world.add(actors['av'])

    world.render()  # show the scenario we created

    return world, actors


def run_scenario(world, time_vector, actors, actions):
    for k in range(0, len(time_vector)):
        for name in actors.keys():
            actors[name].set_control(actions[name][0, k], actions[name][1, k])

        world.tick()  # step the world for one time step
        world.render()  # draw the world
        time.sleep(world.dt / 4)


# Run an example experiment
dt = 0.1  # 100 ms time step
T = 45  # run time seconds
t = np.arange(0, T, dt)  # time vector

# create a scenario
w, actors = create_scenario(dt, world_width=80, world_height=80, pos_intersection=Point(50, 30))

# create action for the actors
# automated vehicle accelerates for the first few seconds, then constant speed. No steering
action_vector = np.zeros((2, t.shape[0]))
action_vector[1, 0:int(10 / dt)] = 0.5
action_vector[1, int(10 / dt):-1] = 0.05  # just a little bit of acceleration to negate friction
actions = {'av': action_vector}

# ego vehicle
# drives up to the stop sign, then turns left
action_vector = np.zeros((2, t.shape[0]))
idx = random.randint(5, 10)
action_vector[0, idx + int(5 / dt):idx + int(11 / dt)] = 0.455
action_vector[1, idx + int(3 / dt):idx + int(12 / dt)] = 0.5
actions.update({'ego': action_vector})

# run the scenario
run_scenario(w, t, actors, actions)
