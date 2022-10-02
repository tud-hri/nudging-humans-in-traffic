"""
Lane classes

Inspiration from D. Sadigh's code
"""

import abc

import casadi
import numpy as np
import pygame

from utils import coordinate_transform


class Lane:
    def __init__(self, p0, p1, width=3.):
        # a lane is straight line between two points (p0 and p1), with a lane width
        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)
        self.width = width
        self.color = (100, 100, 100)
        self.rect = pygame.Rect(0., 0., 1., 1.)

    @abc.abstractmethod
    def draw(self, window, ppm=6):
        pass

    @abc.abstractmethod
    def feature_lane_center(self, c, x):
        pass

    @property
    def x_center_m(self):
        return self.p0[0]

    @property
    def y_center_m(self):
        return self.p0[1]


class HLane(Lane):
    def __init__(self, p0, p1, width):
        super(HLane, self).__init__(p0, p1, width)

    def draw(self, window, ppm=6):
        # transform normal coordinate system (x: right, y: up, bottom left =(0,0)) to graphics coordinates (x: right, y: down, top left = (0,0))
        width = self.width * ppm

        p0 = coordinate_transform(self.p0, ppm)
        p1 = coordinate_transform(self.p1, ppm)

        self.rect.left = min(p0[0], p1[0])
        self.rect.top = min(p0[1], p1[1]) - width / 2.
        self.rect.height = width
        self.rect.width = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

        pygame.draw.rect(window, self.color, self.rect)

    def feature_lane_center(self, x, c=0.25):
        return casadi.exp(-c * (x[1, :] - self.p0[1]) ** 2)

    @property
    def x_bottom_m(self):
        return self.p0[0] - self.width / 2.

    @property
    def x_top_m(self):
        return self.p0[0] + self.width / 2.


class VLane(Lane):
    def __init__(self, p0, p1, width):
        super(VLane, self).__init__(p0, p1, width)

    def draw(self, window, ppm=6):
        width = self.width * ppm

        # transform normal coordinate system (x: right, y: up, bottom left =(0,0)) to graphics coordinates (x: right, y: down, top left = (0,0))
        p0 = coordinate_transform(self.p0, ppm)
        p1 = coordinate_transform(self.p1, ppm)

        self.rect.left = min(p0[0], p1[0]) - width / 2.
        self.rect.top = min(p0[1], p1[1])
        self.rect.height = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        self.rect.width = width

        pygame.draw.rect(window, self.color, self.rect)

        # # draw road lines
        # line_color = (240, 240, 240)
        # pygame.draw.line(window, line_color, (self.rect.left, self.rect.bottom), (self.rect.left, self.rect.top), 1)
        # pygame.draw.line(window, line_color, (self.rect.right, self.rect.bottom), (self.rect.right, self.rect.top), 1)

    def feature_lane_center(self, x, c=0.25):
        return casadi.exp(-c * (x[0, :] - self.p0[0]) ** 2)

    @property
    def x_left_m(self):
        return self.p0[0] - self.width / 2.

    @property
    def x_right_m(self):
        return self.p0[0] + self.width / 2.


class HShoulder:
    def __init__(self, p, side):
        """
        Creates a horizontal shoulder (road boundary)
        :param p: position (x, y) of the shoulder
        :param side: side of the road the shoulder is on ('top', 'bottom')
        """
        self.p = p

        # define sign of sigmoid to determine on what side of the road the shoulder is;
        if side == 'top':
            self._sign = -1.0
        else:
            self._sign = 1.0

    def feature_shoulder(self, x, c=2.):
        """
        Sigmoid function to model the cost of crossing the shoulder (e.g. going off-road: high cost (1), on road: low cost (0))
        :param c: 'steepness' of the sigmoid
        :param x: vector with future states; can be CasADi symbolic parameters
        :return: depending on the type of x, either a np.array or casadi MX/SX
        """
        return 1. / (1. + casadi.exp(- self._sign * c * (self.p[1] - x[1, :])))


class VShoulder:
    def __init__(self, p, side):
        """
        Creates a vertical shoulder (road boundary)
        :param p: position (x, y) of the shoulder
        :param side: side of the road the shoulder is on ('left', 'right')
        """
        self.p = p

        if side == 'left':
            self._sign = 1.0
        else:
            self._sign = -1.0

    def feature_shoulder(self, x, c=4.):
        """
        Sigmoid function to model the cost of crossing the shoulder (e.g. going off-road: high cost (1), on road: low cost (0))
        :param c: 'steepness' of the sigmoid
        :param x: vector with future states; can be CasADi symbolic parameters
        :return: depending on the type of x, either a np.array or casadi MX/SX
        """
        return 1. / (1. + casadi.exp(- self._sign * c * (self.p[0] - x[0, :])))