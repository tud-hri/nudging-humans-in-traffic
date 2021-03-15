import random
import time

from agents import *
from carlo.agents import Painting, RectangleBuilding
from carlo.entities import Point
import carlo.world


class IntersectionWorld(carlo.world.World):
    def __init__(self, dt: float, end_time: float, width: float, height: float, ppm: float = 8, lane_width=3.):
        """
        Redefinition of CARLO's World class, in case we want to customize it
        :param dt:
        :param width:
        :param height:
        :param ppm:
        """
        super(IntersectionWorld, self).__init__(dt, width, height, ppm)

        self.width = width
        self.height = height

        self.play_in_rt = True  # flag if simulation is played in realtime, set to False for quick simulations

        # time vector
        self.time_vector = np.arange(0., end_time, dt)  # time vector

        self.p_intersection = Point(50., 30.)
        self.lane_width = lane_width
        self.create_intersection()

    def create_intersection(self):
        # create the intersection
        n_lanes = 2
        road_width = n_lanes * self.lane_width
        sidewalk_width = 2.

        # in carlo, roads are defined by the sidewalks, so, create sidewalks, buildings
        # this is a bit of a mess
        # bottom left
        self.add(Painting(self.p_intersection / 2. - Point(road_width, road_width) / 2., self.p_intersection, 'gray80'))
        self.add(
            RectangleBuilding(self.p_intersection / 2. - Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2., self.p_intersection, 'forest green'))

        # bottom right
        self.add(Painting(Point(self.p_intersection.x + (self.width - self.p_intersection.x) / 2. + road_width / 2., self.p_intersection.y / 2. - road_width / 2.),
                          Point(self.width - self.p_intersection.x, self.p_intersection.y), 'gray80'))  # sidewalk
        self.add(RectangleBuilding(Point(self.p_intersection.x + (self.width - self.p_intersection.x) / 2. + (road_width + sidewalk_width) / 2.,
                                         self.p_intersection.y / 2. - (road_width + sidewalk_width) / 2.),
                                   Point(self.width - self.p_intersection.x, self.p_intersection.y), 'forest green'))  # building / grass

        # top left
        self.add(Painting(self.p_intersection / 2. + Point(0., self.height / 2.) + Point(-road_width, road_width) / 2,
                          Point(self.p_intersection.x, self.height - self.p_intersection.y), 'gray80'))
        self.add(Painting(self.p_intersection / 2. + Point(0., self.height / 2.) + Point(-(road_width + sidewalk_width), (road_width + sidewalk_width)) / 2,
                          Point(self.p_intersection.x, self.height - self.p_intersection.y), 'forest green'))

        # top right
        pos_topright = Point(self.width, self.height)
        self.add(Painting(pos_topright - (pos_topright - self.p_intersection) / 2. + Point(road_width, road_width) / 2, pos_topright - self.p_intersection, 'gray80'))
        self.add(RectangleBuilding(pos_topright - (pos_topright - self.p_intersection) / 2. + Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2.,
                                   pos_topright - self.p_intersection, 'forest green'))

        # adding lane markings to make things pretty
        lane_marker_length = 2.
        lane_marker_width = 0.2

        # north-south road
        for y in np.arange(0., self.height + 2. * lane_marker_length, 2. * lane_marker_length):
            self.add(Painting(Point(self.p_intersection.x, y), Point(lane_marker_length, lane_marker_width), 'white', heading=np.pi / 2))

        # east-west road
        for x in np.arange(0., self.width + 2. * lane_marker_length, 2. * lane_marker_length):
            self.add(Painting(Point(x, self.p_intersection.y), Point(lane_marker_length, lane_marker_width), 'white', heading=-np.pi))

        # small patch to cover the lane markings at the intersection
        self.add(Painting(self.p_intersection, Point(road_width, road_width), 'gray'))

        # stop lines
        self.add(Painting(self.p_intersection + Point(road_width / 4., -road_width / 2. - 0.15), Point(road_width / 2., 0.3), 'white'))
        self.add(Painting(self.p_intersection + Point(0., -1. - road_width / 2.), Point(0.3, 2.), 'white'))
        self.add(Painting(self.p_intersection + Point(-road_width / 4., road_width / 2. + 0.15), Point(road_width / 2., 0.3), 'white'))
        self.add(Painting(self.p_intersection + Point(0., 1. + road_width / 2.), Point(0.3, 2.), 'white'))
        self.add(Painting(self.p_intersection + Point(road_width / 2. + 0.15, road_width / 4.), Point(road_width / 2., 0.3), 'white', heading=np.pi / 2.))
        self.add(Painting(self.p_intersection + Point(1. + road_width / 2., 0.), Point(0.3, 2.), 'white', heading=np.pi / 2.))
        self.add(Painting(self.p_intersection + Point(-road_width / 2. - 0.15, -road_width / 4.), Point(road_width / 2., 0.3), 'white', heading=np.pi / 2.))
        self.add(Painting(self.p_intersection + Point(-1. - road_width / 2., 0.), Point(0.3, 2.), 'white', heading=np.pi / 2.))

    def run(self):
        for k in range(0, len(self.time_vector)):
            # set control for all agents
            for agent in self.dynamic_agents:
                agent.set_control()

            # step the world and render
            self.tick()

            if self.collision_exists():
                print("Collision")
                # import sys
                # sys.exit(0)

            if self.play_in_rt:
                self.render()  # only render if playing in RT mode
                time.sleep(self.dt / 4.)