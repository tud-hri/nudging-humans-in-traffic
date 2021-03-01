import numpy as np

import carlo.agents
from carlo.entities import Point

from carlo.interactive_controllers import KeyboardController


class Car(carlo.agents.Car):
    def __init__(self, center: Point, heading: float, color: str = 'red'):
        super(Car, self).__init__(center, heading, color)
        self.trajectory = None
        self.u = None


class CarHardCoded(Car):
    def __init__(self, center: Point, heading: float, input, color: str = 'red'):
        super(CarHardCoded, self).__init__(center, heading, color)
        self.u = input
        self.k = 0  # index / time step

    def set_control(self):
        steer = self.u[0, self.k]
        accelerate = self.u[1, self.k]

        super().set_control(steer, accelerate)

        self.k += 1


class CarUserControlled(Car):
    def __init__(self, center: Point, heading: float, color: str = 'blue'):
        super(CarUserControlled, self).__init__(center, heading, color)

    def set_control(self, inputSteering: float, inputAcceleration: float):
        steer = min(max(inputSteering, -np.pi), np.pi)  # limit steer to [-pi, pi]
        accelerate = min(max(inputAcceleration, -4.), 2.) # limit acceleration to [-4., 2.]

        super().set_control(steer, accelerate)


class CarEvidenceAccumulation(Car):
    """
    Car with Arkady's model
    """
    def __init__(self, center: Point, heading: float, color: str = 'green'):
        super(CarEvidenceAccumulation, self).__init__(center, heading, color)