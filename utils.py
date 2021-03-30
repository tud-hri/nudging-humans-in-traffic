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
