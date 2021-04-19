"""
trajectory.py

Stores the trajectory and input of a car
"""
import numpy as np
import pandas as pd


class Trajectory:
    def __init__(self, x0, u0):
        self.data = pd.DataFrame(columns=["t", "x", "y", "orientation", "speed",
                                          "acceleration", "deceleration", "steering_angle"]).set_index("t")
        self.data.loc[0] = np.concatenate((x0.flatten(), u0.flatten()))

        self.x = np.zeros((x0.shape[0], 1))
        self.u = np.zeros((u0.shape[0], 1))
        self.x[:, :0] = x0
        self.u[:, :0] = u0
        self.t = np.zeros((1, 1))

    def append(self, t, x, u):
        self.data.loc[t] = np.concatenate((x.flatten(), u.flatten()))

        self.t = np.append(self.t, np.asarray(t))
        self.x = np.append(self.x, x, axis=1)
        self.u = np.append(self.u, u, axis=1)

    def data_as_dict(self):
        return {"t": self.t, "x": self.x, "u": self.u}

    def data_as_array(self):
        # FIXME: this currently throws an exception - t, x, and u must have the same number of dimensions
        # Perhaps this is not needed if we use pandas for storing the data in the first place?
        return np.concatenate((self.t, self.x, self.u), axis=0)

    @property
    def x0(self):
        if self.x.ndim == 1:
            return self.x
        else:
            return self.x[:, 0]
