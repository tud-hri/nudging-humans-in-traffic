'''
Our world: an intersection with left-turn scenario
'''

import time

import numpy as np

from carlo.agents import Car, Painting, RectangleBuilding
from carlo.geometry import Point
from carlo.world import World

dt = 0.01  # 10 ms time step


def create_scenario(time_step):
    n_lanes = 2
    lane_width = 3  # meters
    road_width = n_lanes * lane_width
    world_width = 80  # meters
    world_height = 160  # meters
    sidewalk_width = 2

    actors = {}

    # intersection location
    pos_intersection = Point(world_width / 2, world_height / 3)

    # create our world

    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0, rad)
    w = World(dt, width=world_width, height=world_height, ppm=6)  # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # in carlo, roads are defined by the sidewalks, so, create sidewalks, buildings

    # this is a bit of a mess

    # bottom left
    w.add(Painting(pos_intersection / 2 - Point(road_width, road_width) / 2, pos_intersection, 'gray80'))
    w.add(RectangleBuilding(pos_intersection / 2 - Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2, pos_intersection, 'forest green'))

    # bottom right
    w.add(Painting(Point(pos_intersection.x + (world_width - pos_intersection.x) / 2 + road_width / 2, pos_intersection.y / 2 - road_width / 2),
                   Point(world_width - pos_intersection.x, pos_intersection.y), 'gray80'))  # sidewalk
    w.add(RectangleBuilding(Point(pos_intersection.x + (world_width - pos_intersection.x) / 2 + (road_width + sidewalk_width) / 2,
                                  pos_intersection.y / 2 - (road_width + sidewalk_width) / 2),
                            Point(world_width - pos_intersection.x, pos_intersection.y), 'forest green'))  # building / grass

    # top left
    w.add(Painting(pos_intersection / 2 + Point(0, world_height / 2) + Point(-road_width, road_width) / 2,
                   Point(pos_intersection.x, world_height - pos_intersection.y), 'gray80'))
    w.add(Painting(pos_intersection / 2 + Point(0, world_height / 2) + Point(-(road_width + sidewalk_width), (road_width + sidewalk_width)) / 2,
                   Point(pos_intersection.x, world_height - pos_intersection.y), 'forest green'))

    # top right
    pos_topright = Point(world_width, world_height)
    w.add(Painting(pos_topright - (pos_topright - pos_intersection) / 2 + Point(road_width, road_width) / 2, pos_topright - pos_intersection, 'gray80'))
    w.add(RectangleBuilding(pos_topright - (pos_topright - pos_intersection) / 2 + Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2,
                            pos_topright - pos_intersection, 'forest green'))

    # adding lane markings to make things pretty
    lane_marker_length = 2
    lane_marker_width = 0.2

    # north-south road
    for y in np.arange(0, world_height + 2 * lane_marker_length, 2 * lane_marker_length):
        w.add(Painting(Point(pos_intersection.x, y), Point(lane_marker_length, lane_marker_width), 'white', heading=np.pi / 2))

    # east-west road
    for x in np.arange(0, world_width + 2 * lane_marker_length, 2 * lane_marker_length):
        w.add(Painting(Point(x, pos_intersection.y), Point(lane_marker_length, lane_marker_width), 'white', heading=- np.pi))

    # small patch to cover the lane markings at the intersection
    w.add(Painting(pos_intersection, Point(road_width, road_width), 'gray'))

    # stop lines
    w.add(Painting(pos_intersection + Point(road_width / 4, -road_width / 2 - 0.15), Point(road_width / 2, 0.3),'white'))
    w.add(Painting(pos_intersection + Point(0, -1 - road_width/2), Point(0.3,2), 'white'))
    w.add(Painting(pos_intersection + Point(-road_width / 4, road_width / 2 + 0.15), Point(road_width / 2, 0.3), 'white'))
    w.add(Painting(pos_intersection + Point(0, 1 + road_width / 2), Point(0.3, 2), 'white'))
    w.add(Painting(pos_intersection + Point(road_width / 2 + 0.15, road_width / 4), Point(road_width / 2, 0.3), 'white', heading=np.pi / 2))
    w.add(Painting(pos_intersection + Point(1 + road_width / 2, 0), Point(0.3,2), 'white', heading=np.pi / 2))
    w.add(Painting(pos_intersection + Point(-road_width / 2 - 0.15, -road_width / 4), Point(road_width / 2, 0.3), 'white', heading=np.pi / 2))
    w.add(Painting(pos_intersection + Point(-1 - road_width / 2, 0), Point(0.3,2), 'white', heading=np.pi / 2))

    # add ego vehicle
    actors['ego'] = Car(Point(pos_intersection.x + road_width / 4, 5), np.pi / 2, 'blue')
    w.add(actors['ego'])

    # add AV
    actors['av'] = Car(Point(pos_intersection.x - road_width / 4, world_height - 5), - np.pi / 2)
    w.add(actors['av'])

    w.render()  # show the scenario we created

    return w, actors


def run_scenario():
    pass  # for now


w = create_scenario(dt)

time.sleep(5)
