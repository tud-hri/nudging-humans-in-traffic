import numpy as np
import pygame
from pygame.locals import *

from dynamics import CarDynamics
from utils import coordinate_transform


class Car:
    def __init__(self, p0, phi0: float, v0: float = 0., dt: float = 0.1, color: str = 'red'):
        x0 = np.array([p0[0], p0[1], phi0, v0])  # initial condition
        self.dt = dt
        self.dynamics = CarDynamics(dt, x0=x0)
        # self.trajectory = None # work in progress - at some point we might want to store state and input in a separate object (and the histories)
        self.u = np.zeros((2, 1))  # [acceleration, steering]
        self.x = x0  # [x, y, phi, v]
        self.world = None
        self.car_width = 2.  # width of the car
        self.car_length = self.dynamics.length

        self.image = pygame.image.load("img/car-{0}.png".format(color))

    def set_input(self, accelerate: float = 0., steer: float = 0.):
        """
        :param accelerate:
        :param steer:
        :return:
        """
        self.u[0] = accelerate
        self.u[1] = steer

    def tick(self):
        """
        Perform one integration step of the dynamics
        This is an override of the Entity.tick function, to enable us to define our own dynamics (e.g., as a CasADi function)
        :param dt:
        :return: car state
        """
        x_next = self.dynamics.integrate(self.x, self.u)
        self.x = x_next.full()  # casadi DM to np.array

    def draw(self, window, ppm):
        # coordinate transform to graphics coordinate frame
        p = self.x[0:2] * ppm
        p = coordinate_transform(p)

        img = pygame.transform.scale(self.image, (int(self.car_length * ppm), int(self.car_width * ppm)))
        img = pygame.transform.rotate(img, np.rad2deg(self.x[2]))

        # calculate center position for drawing
        img_rect = img.get_rect()
        img_rect.center = (p[0, 0], p[1, 0])

        window.blit(img, img_rect)


class CarUserControlled(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., dt: float = 0.1, color: str = 'blue'):
        super(CarUserControlled, self).__init__(p0, phi0, v0, dt, color)
        self.accelerate = 0.
        self.steer = 0.

    def set_input(self, accelerate=0., steer=0.):
        """
        Crappy implementation of a simple keyboard controller
        :param accelerate:
        :param steer:
        :return:
        """
        accelerate_sensitivity = 2  # [m/s2 / s]
        decelerate_sensitivity = 3
        steer_sensitivity = 1.5 * np.pi  # [rad/s]

        keys = pygame.key.get_pressed()

        if not keys[K_UP] or not keys[K_DOWN]:
            self.accelerate = 0.
        if keys[K_UP]:
            self.accelerate += accelerate_sensitivity * self.dt
        elif keys[K_DOWN]:
            self.accelerate -= decelerate_sensitivity * self.dt
        accelerate = min(max(self.accelerate, -4.), 2.)  # limit acceleration to [-4., 2.]

        if not keys[K_LEFT] or not keys[K_RIGHT]:
            self.steer = 0.
        if keys[K_LEFT]:
            self.steer += steer_sensitivity * self.dt
        elif keys[K_RIGHT]:
            self.steer -= steer_sensitivity * self.dt
        steer = min(max(self.steer, -np.pi), np.pi)  # limit steer to [-pi, pi]

        super().set_input(accelerate, steer)

# class CarHardCoded(Car):
#     def __init__(self, center: Point, heading: float, input, dt: float = 0.1, color: str = 'red'):
#         super(CarHardCoded, self).__init__(center, heading, dt, color)
#         self.u = input
#         self.k = 0  # index / time step
#
#     def set_control(self):
#         steer = self.u[0, self.k]
#         accelerate = self.u[1, self.k]
#
#         super().set_control(steer, accelerate)
#
#         self.k += 1
#
#
# class CarUserControlled(Car):
#     def __init__(self, center: Point, heading: float, dt: float = 0.1, color: str = 'blue'):
#         super(CarUserControlled, self).__init__(center, heading, dt, color)
#         self.controller = None
#
#     def set_control(self):
#         if self.controller is None:
#             self.controller = KeyboardController(self.world)
#
#         steer = min(max(self.controller.steering, -np.pi), np.pi)  # limit steer to [-pi, pi]
#         accelerate = min(max(self.controller.throttle, -4.), 2.)  # limit acceleration to [-4., 2.]
#
#         super().set_control(steer, accelerate)
#
# class CarEvidenceAccumulation(Car):
#     """
#     Car with Arkady's model
#     """
#
#     def __init__(self, center: Point, heading: float, dt: float = 0.1, color: str = 'green'):
#         super(CarEvidenceAccumulation, self).__init__(center, heading, dt, color)
