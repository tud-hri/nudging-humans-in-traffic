"""
trajectory.py

Stores the trajectory and input of a car
"""
import numpy as np


class Trajectory:

    def __init__(self, x0, u0):
        self.x = np.asarray(x0)
        self.u = np.asarray(u0)
        self.t = [0.]

    def append(self, t, x, u):
        self.t.append(t)
        np.append(self.x, x)
        np.append(self.u, u)

    @property
    def x0(self):
        if self.x.ndim == 1:
            return self.x
        else:
            return self.x[:, 0]
