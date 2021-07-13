import numpy as np
import pygame

from lane import VLane, HLane, VShoulder, HShoulder
from utils import coordinate_transform


class IntersectionWorld:
    def __init__(self, dt: float, width: float, height: float, show_state_text=True):
        self.dt = dt
        self.width = width  # [m]
        self.height = height  # [m]

        self.agents = {}
        self.lanes = []
        self.lane_width = 3.5
        self.shoulders = []  # all the road shoulders, added to keep the cars on the road.
        self.p_intersection = np.array([40., 30.])

        self.p_pet_square = np.array([self.p_intersection[0] - self.lane_width / 2., self.p_intersection[1] + self.lane_width / 2.])
        self.pet_rect = pygame.Rect(self.p_pet_square[0], self.p_pet_square[1], self.lane_width * 10, self.lane_width * 10)

        self.collision = []
        self.create_intersection()
        self.show_state_text = show_state_text

        self.human_in_pet_zone_prev = False
        self.av_in_pet_zone_prev = False
        self.t_pet_out_human = np.NAN
        self.t_pet_out_av = np.NAN


    def reset(self):

        # reset the post encroachment time parameters
        self.human_in_pet_zone_prev = False
        self.av_in_pet_zone_prev = False
        self.t_pet_out_human = np.NAN
        self.t_pet_out_av = np.NAN

        # reset collision list
        self.collision = []

    def create_intersection(self):
        # lanes
        self.lanes.append(VLane([self.p_intersection[0] - self.lane_width / 2., 0.],
                                [self.p_intersection[0] - self.lane_width / 2., 120.], self.lane_width))
        self.lanes.append(VLane([self.p_intersection[0] + self.lane_width / 2., 0.],
                                [self.p_intersection[0] + self.lane_width / 2., 120.], self.lane_width))

        self.lanes.append(HLane([0., self.p_intersection[1] - self.lane_width / 2.],
                                [self.p_intersection[0], self.p_intersection[1] - self.lane_width / 2.], self.lane_width))
        self.lanes.append(HLane([0., self.p_intersection[1] + self.lane_width / 2.],
                                [self.p_intersection[0], self.p_intersection[1] + self.lane_width / 2.], self.lane_width))

        # shoulders / bounds
        self.shoulders.append(HShoulder([0., self.p_intersection[1]], side='top'))  # shoulder left turn, top
        self.shoulders.append(VShoulder([self.p_intersection[1] + self.lane_width, 0.], side='left'))  # shoulder left of vertical road
        self.shoulders.append(VShoulder([self.p_intersection[0] + self.lane_width, 0.], side='right'))  # shoulder right of vertical road

    def tick(self, sim_time: float, step: int):
        # find action
        for agent in self.agents.values():
            agent.calculate_action(sim_time, step)

        # apply action, integrate
        for agent in self.agents.values():
            agent.tick(sim_time, step)

        # check for collisions among all agents
        self.collision = [a1.rect.colliderect(a2.rect) for a1 in self.agents.values() for a2 in self.agents.values() if a1 is not a2]

        # calculate post encroachment time
        # PET is calculate by finding the time when the left turning car exits the intersection and the AV enters the intersection.
        # we do this through pygame's collision detection with a invisible rectangle
        human = self.agents["human"]
        if human.decision == "go":
            collision_detected = human.rect.colliderect(self.pet_rect)
            if not collision_detected and self.human_in_pet_zone_prev:
                # means the human is out of the pet zone
                self.t_pet_out_human = sim_time
                # print("t_pet_out_human: " + str(self.t_pet_out_human))
            self.human_in_pet_zone_prev = collision_detected

        av = self.agents["av"]
        collision_detected = av.rect.colliderect(self.pet_rect)
        if collision_detected and not self.av_in_pet_zone_prev:
            # means the av enters the pet zone
            self.t_pet_out_av = sim_time
            # print("t_pet_out_av: " + str(self.t_pet_out_av))
        self.av_in_pet_zone_prev = collision_detected

    def draw(self, window, ppm):
        window.fill((33, 138, 33))

        for lane in self.lanes:
            lane.draw(window, ppm)

        # stop lines
        line_color = (240, 240, 240)
        p0 = coordinate_transform(np.array([self.p_intersection[0] - self.lane_width, self.p_intersection[1]]), ppm)
        p1 = coordinate_transform(np.array([self.p_intersection[0] - self.lane_width, self.p_intersection[1] - self.lane_width]), ppm)
        pygame.draw.line(window, line_color, tuple(p0), tuple(p1), 1)

        # dashed lines vertical road
        p_start = coordinate_transform(np.array([self.p_intersection[0], self.height]), ppm)
        p_end = coordinate_transform(np.array([self.p_intersection[0], 0.]), ppm)
        points = np.arange(p_start[1], p_end[1], 1.5 * ppm)
        for ii in range(0, len(points) - 1, 2):
            pygame.draw.line(window, line_color, (p_start[0], points[ii]), (p_end[0], points[ii + 1]), 1)

        # # dashed lines horizontal road
        p_start = coordinate_transform(np.array([0., self.p_intersection[1]]), ppm)
        p_end = coordinate_transform(np.array([self.p_intersection[0] - self.lane_width, self.p_intersection[1]]), ppm)
        points = np.arange(p_start[0], p_end[0], 1.5 * ppm)
        for ii in range(0, len(points) - 1, 2):
            pygame.draw.line(window, line_color, (points[ii], p_start[1]), (points[ii + 1], p_start[1]), 1)

        # draw agents
        pos = (5, 5)
        for agent in self.agents.values():
            agent.draw(window, ppm)
            if self.show_state_text:
                state_text = agent.text_state_render()
                window.blit(state_text, state_text.get_rect(left=pos[0], top=pos[1]))
                pos = (pos[0], state_text.get_rect().bottom + 0.25 * state_text.get_height())

        # update pet square to pygame coordinates
        p = coordinate_transform(self.p_pet_square + np.array([-self.lane_width, self.lane_width]) / 2., ppm)
        self.pet_rect = pygame.Rect(p[0], p[1], 0.8 * self.lane_width * ppm, 0.8 * self.lane_width * ppm)

        # pygame.draw.rect(window, (255,0,0), self.pet_rect)
