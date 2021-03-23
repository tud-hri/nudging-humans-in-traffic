"""
Lane classes

Inspiration from D. Sadigh's code
"""

import abc

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

    @abc.abstractmethod
    def draw(self, window, ppm=6):
        pass

    # @abc.abstractmethod
    # def boundary(self, x, phi, d=10, n=10):
    #     """
    #     return the left and right road boundaries for a lookahead distance l as seen from position x, for a number of samples N.
    #     :param x: current position (2,1) [m, m]
    #     :param phi: current heading (1,1) [rad]
    #     :param d: look ahead distance
    #     :param n: number of samples
    #     :return: nparray (2,N) with left and right boundaries (car's perspective)
    #     """
    #     pass


class HLane(Lane):
    def __init__(self, p0, p1, width):
        super(HLane, self).__init__(p0, p1, width)

    def draw(self, window, ppm=6):
        # transform normal coordinate system (x: right, y: up, bottom left =(0,0)) to graphics coordinates (x: right, y: down, top left = (0,0))
        p0 = self.p0 * ppm
        p1 = self.p1 * ppm
        width = self.width * ppm

        p0 = coordinate_transform(p0)
        p1 = coordinate_transform(p1)

        # find top-left corner and width, length
        x = min(p0[0], p1[0])
        y = min(p0[1], p1[1]) - width / 2.
        length = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        rect = pygame.Rect(x, y, length, width)

        pygame.draw.rect(window, self.color, rect)

        # draw road lines
        line_color = (240, 240, 240)
        pygame.draw.line(window, line_color, (rect.left, rect.bottom), (rect.right, rect.bottom), 1)
        pygame.draw.line(window, line_color, (rect.left, rect.top), (rect.right, rect.top), 1)


class VLane(Lane):
    def __init__(self, p0, p1, width):
        super(VLane, self).__init__(p0, p1, width)

    def draw(self, window, ppm=6):
        p0 = self.p0 * ppm
        p1 = self.p1 * ppm
        width = self.width * ppm

        # transform normal coordinate system (x: right, y: up, bottom left =(0,0)) to graphics coordinates (x: right, y: down, top left = (0,0))
        p0 = coordinate_transform(p0)
        p1 = coordinate_transform(p1)

        x = min(p0[0], p1[0]) - width / 2.
        y = min(p0[1], p1[1])
        length = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)

        rect = pygame.Rect(x, y, width, length)
        pygame.draw.rect(window, self.color, rect)

        # draw road lines
        line_color = (240, 240, 240)
        pygame.draw.line(window, line_color, (rect.left, rect.bottom), (rect.left, rect.top), 1)
        pygame.draw.line(window, line_color, (rect.right, rect.bottom), (rect.right, rect.top), 1)