import numpy as np

import carlo
from carlo.entities import Point


class Car(carlo.agents.Car):
    def __init__(self, center: Point, heading: float, color: str = 'red'):
        super(Car, self).__init__(center, heading, color)
        self.trajectory = None
        self.u = None


class CarHardCoded(Car):
    def __init__(self, center: Point, heading: float, input, color: str = 'red'):
        super(CarHardCoded, self).__init__(center, heading, color)
        self.u = input

    def set_control(self, k: int):
        steer = self.u[0, k]
        accelerate = self.u[1, k]

        super().set_control(steer, accelerate)


class CarUserControlled(Car):
    def __init__(self, center: Point, heading: float, color: str = 'blue'):
        super(CarHardCoded, self).__init__(center, heading, color)

    def set_control(self, steer, accelerate):
        steer = min(max(steer, -np.pi), np.pi)  # limit steer to [-pi, pi]
        accelerate = min(max(accelerate, -4.), 2.) # limit acceleration to [-4., 2.]

        self.set_control(steer, accelerate)

class CarEvidenceAccumulation(Car):
    """
    Car with Arkady's model
    """
    def __init__(self, center: Point, heading: float, color: str = 'green'):
        super(CarEvidenceAccumulation, self).__init__(center, heading, color)