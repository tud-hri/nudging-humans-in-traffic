import numpy as np

import carlo.agents
from carlo.entities import Point
from carlo.interactive_controllers import KeyboardController
from dynamics import CarDynamics


class Car(carlo.agents.Car):
    def __init__(self, center: Point, heading: float, dt: float = 0.1, color: str = 'red'):
        super(Car, self).__init__(center, heading, color)
        x0 = np.array([center.x, center.y, heading, 0.])  # initial condition
        self.dynamics = CarDynamics(dt, x0=x0)
        # self.trajectory = None # work in progress - at some point we might want to store state and input in a separate object (and the histories)
        self.u_input = np.zeros((2, 1))  # [acceleration, steering]
        self.x_state = x0  # [x, y, phi, v]
        self.world = None

    def set_control(self, inputSteering: float, inputAcceleration: float):
        """
        Override from CARLO: set self.u_input
        :param inputSteering:
        :param inputAcceleration:
        :return:
        """
        self.inputSteering = inputSteering
        self.inputAcceleration = inputAcceleration

        self.u_input[0] = self.inputAcceleration
        self.u_input[1] = self.inputSteering

    def tick(self, dt: float):
        """
        Perform one integration step of the dynamics
        This is an override of the Entity.tick function, to enable us to define our own dynamics (e.g., as a CasADi function)
        :param dt:
        :return: car state
        """
        x_next = self.dynamics.integrate(self.x_state, self.u_input)
        self.x_state = x_next

        # and convert state to Point for CARLO
        self.center = Point(x_next[0], x_next[1])
        self.heading = np.mod(x_next[2], 2 * np.pi)  # wrap the heading angle between 0 and +2pi
        self.velocity = x_next[3]


class CarHardCoded(Car):
    def __init__(self, center: Point, heading: float, input, dt: float = 0.1, color: str = 'red'):
        super(CarHardCoded, self).__init__(center, heading, dt, color)
        self.u = input
        self.k = 0  # index / time step

    def set_control(self):
        steer = self.u[0, self.k]
        accelerate = self.u[1, self.k]

        super().set_control(steer, accelerate)

        self.k += 1


class CarUserControlled(Car):
    def __init__(self, center: Point, heading: float, dt: float = 0.1, color: str = 'blue'):
        super(CarUserControlled, self).__init__(center, heading, dt, color)
        self.controller = None

    def set_control(self):
        if self.controller is None:
            self.controller = KeyboardController(self.world)

        steer = min(max(self.controller.steering, -np.pi), np.pi)  # limit steer to [-pi, pi]
        accelerate = min(max(self.controller.throttle, -4.), 2.)  # limit acceleration to [-4., 2.]

        super().set_control(steer, accelerate)

class CarEvidenceAccumulation(Car):
    """
    Car with Arkady's model
    """

    def __init__(self, center: Point, heading: float, dt: float = 0.1, color: str = 'green'):
        super(CarEvidenceAccumulation, self).__init__(center, heading, dt, color)
