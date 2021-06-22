import numpy as np
import pygame

from lane import VLane, HLane, VShoulder, HShoulder
from utils import coordinate_transform


class IntersectionWorld:
    def __init__(self, dt: float, width: float, height: float, show_state_text=True):
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
        self.shoulders = []  # all the road shoulders, added to keep the cars on the road.

        self.collision = []
        self.create_intersection()
        self.show_state_text = show_state_text

    def create_intersection(self):

        # lanes
        self.lanes.append(HLane([0., 30.], [40., 30.], 3.))
        self.lanes.append(HLane([0., 27.], [40., 27.], 3.))
        self.lanes.append(VLane([40., 0.], [40., 120.], 3.))
        self.lanes.append(VLane([37., 0.], [37., 120.], 3.))

        # shoulders / bounds
        self.shoulders.append(HShoulder([0., 32], side='top'))  # shoulder left turn, top
        self.shoulders.append(VShoulder([35, 0.], side='left'))  # shoulder left of vertical road
        self.shoulders.append(VShoulder([42, 0.], side='right'))  # shoulder right of vertical road

    def tick(self, sim_time: float, step: int):
        # find action
        for agent in self.agents.values():
            agent.calculate_action(sim_time, step)

        # apply action, integrate
        for agent in self.agents.values():
            agent.tick(sim_time, step)

        # check for collisions among all agents
        self.collision = [a1.rect.colliderect(a2.rect) for a1 in self.agents.values() for a2 in self.agents.values() if a1 is not a2]

    def draw(self, window, ppm):
        window.fill((33, 138, 33))

        for lane in self.lanes:
            lane.draw(window, ppm)

        line_color = (240, 240, 240)
        p0 = coordinate_transform(np.array([38.5 * ppm, 25.5 * ppm]))
        p1 = coordinate_transform(np.array([41.5 * ppm, 25.5 * ppm]))
        pygame.draw.line(window, line_color, tuple(p0), tuple(p1), 1)
        # draw lane lines
        # line_color = (240, 240, 240)
        # pygame.draw.line(window, line_color, (self.rect.left, self.rect.bottom), (self.rect.right, self.rect.bottom), 1)
        # pygame.draw.line(window, line_color, (self.rect.left, self.rect.top), (self.rect.right, self.rect.top), 1)

        # draw agents
        state_text = []
        pos = (5, 5)
        for agent in self.agents.values():
            agent.draw(window, ppm)
            if self.show_state_text:
                state_text = agent.text_state_render()
                window.blit(state_text, state_text.get_rect(left=pos[0], top=pos[1]))
                pos = (pos[0], state_text.get_rect().bottom + 0.25 * state_text.get_height())
