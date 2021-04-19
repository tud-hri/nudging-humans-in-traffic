"""
utils.py
Static functions for common utilities
"""

import casadi
import numpy as np
import pygame


def coordinate_transform(p):
    _, h = pygame.display.get_surface().get_size()
    return np.array([p[0], h - p[1]])


def rotmatrix_casadi(phi):
    R = casadi.MX(2, 2)
    R[0, 0] = casadi.cos(phi)
    R[0, 1] = -casadi.sin(phi)
    R[1, 0] = casadi.sin(phi)
    R[1, 1] = casadi.cos(phi)
    return R


def rotmatrix(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def get_derivative(t, x):
    # To be able to reasonably calculate derivatives at the end-points of the trajectories,
    # append three extra points before and after the actual trajectory, so we get N+6
    # points instead of N
    # TODO: make a linear extrapolation instead of constant
    x = np.append(x[0] * np.ones(3), np.append(x, x[-1] * np.ones(3)))

    # Time vector is also artificially extended by equally spaced points
    # Use median timestep to add dummy points to the time vector
    timestep = np.median(np.diff(t))
    t = np.append(t[0] - np.arange(1, 4) * timestep, np.append(t, t[-1] + np.arange(1, 4) * timestep))

    # smooth noise-robust differentiators, see:
    # http://www.holoborodko.com/pavel/numerical-methods/ \
    # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
    v = (1 * (x[6:] - x[:-6]) / ((t[6:] - t[:-6]) / 6) +
         4 * (x[5:-1] - x[1:-5]) / ((t[5:-1] - t[1:-5]) / 4) +
         5 * (x[4:-2] - x[2:-4]) / ((t[4:-2] - t[2:-4]) / 2)) / 32

    return v
