import numpy as np
import pygame
from pygame.locals import *

import scenarios
from intersection_world import IntersectionWorld


class Simulator:
    def __init__(self, world, T: float, dt: float = 0.1, ppm: int = 6, realtime=True):
        self.T = T
        self.dt = max(dt, 0.02)  # max freq = 50Hz
        self.N = round(self.T / self.dt)
        self.t = np.linspace(0., self.T - self.dt, self.N)
        self.world = world
        self.ppm = ppm  # pixel per meter
        self.realtime = realtime  # should the sim go 'soft' real time or AFAP?

        # setup pygame, create a window if we're going to visualize things
        pygame.init()
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((int(self.world.width * self.ppm), int(self.world.height * self.ppm)))
        pygame.display.set_caption('Mind-reading AVs')
        self.window.fill((33, 138, 33))  # green background
        pygame.display.flip()

        self.font_state = pygame.font.SysFont("verdana", 12)

    def run(self):
        running = True
        t0 = pygame.time.get_ticks()
        counter = 0

        while running:
            if self.realtime:
                self.clock.tick(round(1. / self.dt))  # realtime, so wait a bit for the next tick/frame

            # do all the functional stuff here
            self.world.tick(self.t[counter])

            # time keeping

            t_elapsed = (pygame.time.get_ticks() - t0) * 1e-3

            # draw the world
            self.world.draw(self.window, self.ppm)

            # simulator state
            sim_state_text = "t_sim: {:.2f}".format(self.t[counter])
            text = self.font_state.render(sim_state_text, True, (0, 0, 0))
            self.window.blit(text, text.get_rect(left=5, bottom=self.window.get_height() - 5))  # center=self.window.get_rect().center

            pygame.display.flip()

            counter += 1

            if self.dt * counter >= self.T:
                running = False
                print("Time's up, we're done here. Simulation finished in {0} seconds".format(t_elapsed))

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False


if __name__ == '__main__':
    # a whole new woooooorld
    world = IntersectionWorld(0.1, 80., 100.)

    scenario1 = scenarios.scenario1(world)

    sim = Simulator(scenario1, T=20., dt=0.1)
    sim.run()
