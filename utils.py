"""
utils.py
Static functions for common utilities
"""

import pygame
import numpy as np


def coordinate_transform(p):
    _, h = pygame.display.get_surface().get_size()
    return np.array([p[0], h - p[1]])
