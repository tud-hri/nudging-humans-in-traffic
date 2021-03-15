from lane import Lane


class World:
    def __init__(self, dt: float, width: float, height: float):
        """
        The world our agents live in
        :param dt: simulation time step [second]
        :param width: world width [meter]
        :param height: world height [meter]
        """
        self.dt = dt
        self.width = width  # [m]
        self.height = height  # [m]

        self.agents = {}
        self.lanes = []

    def tick(self):
        pass
        # find action
        # for _, agent in self.agents.items():
        #     agent.set_control()

        # apply action, integrate
        # for _, agent, in self.agents.items():
        #     agent.tick()

    def draw(self, window, ppm):
        for lane in self.lanes:
            lane.draw(window, ppm)

        for _, agent in self.agents.items():
            agent.draw(window, ppm)

#     def create_intersection(self, p_intersection=Point(50., 30.), lane_width=3.):
#         # create the intersection
#         n_lanes = 2
#         road_width = n_lanes * lane_width
#         sidewalk_width = 2.
#
#         # in carlo, roads are defined by the sidewalks, so, create sidewalks, buildings
#         # this is a bit of a mess
#         # bottom left
#         self.add(Painting(p_intersection / 2. - Point(road_width, road_width) / 2., p_intersection, 'gray80'))
#         self.add(
#             RectangleBuilding(p_intersection / 2. - Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2., p_intersection, 'forest green'))
#
#         # bottom right
#         self.add(Painting(Point(p_intersection.x + (self.width - p_intersection.x) / 2. + road_width / 2., p_intersection.y / 2. - road_width / 2.),
#                           Point(self.width - p_intersection.x, p_intersection.y), 'gray80'))  # sidewalk
#         self.add(RectangleBuilding(Point(p_intersection.x + (self.width - p_intersection.x) / 2. + (road_width + sidewalk_width) / 2.,
#                                          p_intersection.y / 2. - (road_width + sidewalk_width) / 2.),
#                                    Point(self.width - p_intersection.x, p_intersection.y), 'forest green'))  # building / grass
#
#         # top left
#         self.add(Painting(p_intersection / 2. + Point(0., self.height / 2.) + Point(-road_width, road_width) / 2,
#                           Point(p_intersection.x, self.height - p_intersection.y), 'gray80'))
#         self.add(Painting(p_intersection / 2. + Point(0., self.height / 2.) + Point(-(road_width + sidewalk_width), (road_width + sidewalk_width)) / 2,
#                           Point(p_intersection.x, self.height - p_intersection.y), 'forest green'))
#
#         # top right
#         pos_topright = Point(self.width, self.height)
#         self.add(Painting(pos_topright - (pos_topright - p_intersection) / 2. + Point(road_width, road_width) / 2, pos_topright - p_intersection, 'gray80'))
#         self.add(RectangleBuilding(pos_topright - (pos_topright - p_intersection) / 2. + Point(road_width + sidewalk_width, road_width + sidewalk_width) / 2.,
#                                    pos_topright - p_intersection, 'forest green'))
#
#         # adding lane markings to make things pretty
#         lane_marker_length = 2.
#         lane_marker_width = 0.2
#
#         # north-south road
#         for y in np.arange(0., self.height + 2. * lane_marker_length, 2. * lane_marker_length):
#             self.add(Painting(Point(p_intersection.x, y), Point(lane_marker_length, lane_marker_width), 'white', heading=np.pi / 2))
#
#         # east-west road
#         for x in np.arange(0., self.width + 2. * lane_marker_length, 2. * lane_marker_length):
#             self.add(Painting(Point(x, p_intersection.y), Point(lane_marker_length, lane_marker_width), 'white', heading=-np.pi))
#
#         # small patch to cover the lane markings at the intersection
#         self.add(Painting(p_intersection, Point(road_width, road_width), 'gray'))
#
#         # stop lines
#         self.add(Painting(p_intersection + Point(road_width / 4., -road_width / 2. - 0.15), Point(road_width / 2., 0.3), 'white'))
#         self.add(Painting(p_intersection + Point(0., -1. - road_width / 2.), Point(0.3, 2.), 'white'))
#         self.add(Painting(p_intersection + Point(-road_width / 4., road_width / 2. + 0.15), Point(road_width / 2., 0.3), 'white'))
#         self.add(Painting(p_intersection + Point(0., 1. + road_width / 2.), Point(0.3, 2.), 'white'))
#         self.add(Painting(p_intersection + Point(road_width / 2. + 0.15, road_width / 4.), Point(road_width / 2., 0.3), 'white', heading=np.pi / 2.))
#         self.add(Painting(p_intersection + Point(1. + road_width / 2., 0.), Point(0.3, 2.), 'white', heading=np.pi / 2.))
#         self.add(Painting(p_intersection + Point(-road_width / 2. - 0.15, -road_width / 4.), Point(road_width / 2., 0.3), 'white', heading=np.pi / 2.))
#         self.add(Painting(p_intersection + Point(-1. - road_width / 2., 0.), Point(0.3, 2.), 'white', heading=np.pi / 2.))
#
#
# def scenario1(dt, end_time):
#     """
#     Scenario 1
#     Straight intersection, two approaching cars, ego vehicle turns left, automated vehicle goes straight
#     :param dt: time step
#     :param end_time: end time of simulation
#     """
#
#     # create our world
#     # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
#     world = World(dt, end_time, width=80., height=80., ppm=6.)  # The world is 120 meters by 120 meters. ppm is the pixels per meter.
#     p_intersection = Point(50., 30.)
#     lane_width = 3.
#     world.create_intersection(p_intersection=p_intersection, lane_width=lane_width)  # create the intersection
#
#     # cars
#     # add ego vehicle (hardcoded inputs)
#     # at stop sign at the intersection, accelerated and turns left
#     u = np.zeros((2, world.time_vector.shape[0]))
#     idx = random.randint(5, 10)
#     u[0, idx + int(5 / dt):idx + int(11 / dt)] = 0.455
#     u[1, idx + int(3 / dt):idx + int(12 / dt)] = 0.5
#     # car_ego = CarHardCoded(center=Point(p_intersection.x + 2 * lane_width / 4., p_intersection.y - 2 * lane_width / 2. - 3.), heading=np.pi / 2., world=world,
#     #                        input=u, color='blue')
#     car_ego = CarUserControlled(center=Point(p_intersection.x + 2 * lane_width / 4., p_intersection.y - 2 * lane_width / 2. - 3.), heading=np.pi / 2.,
#                                 color='yellow')
#     world.add(car_ego)
#
#     # add AV (hardcoded inputs)
#     u = np.zeros((2, world.time_vector.shape[0]))
#     u[1, 0:int(10 / dt)] = 0.5
#     u[1, int(10 / dt):-1] = 0.05  # just a little bit of acceleration to negate friction
#     car_av = CarHardCoded(center=Point(p_intersection.x - 2 * lane_width / 4., world.height - 25.), heading=- np.pi / 2., input=u)
#     world.add(car_av)
#
#     # render, just to get to see our work come to life
#     world.render()
#
#     # @Arkady - my attempt, but this is suboptimal, want to use a @setter for this ideally
#     # giving the cars 'world' as a paremeter doesn't seem to work for the usercontrolled car.
#     car_ego.world = world
#     car_av.world = world
#
#     return world
