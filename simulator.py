import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame.locals import *

import scenarios
from intersection_world import IntersectionWorld


class Simulator:
    def __init__(self, world, end_time: float, dt: float = 0.1, ppm: int = 6, realtime=True):
        self.T = end_time
        self.dt = world.dt  # max freq = 50Hz
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

    def run(self):
        running = True
        paused = False
        kill_switch_pressed = False
        t0 = pygame.time.get_ticks()
        counter = 0

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                        kill_switch_pressed = True
                        return kill_switch_pressed
                    if event.key == K_SPACE:
                        paused = not paused

            if paused:
                continue

            if self.realtime:
                self.clock.tick(round(1. / self.dt))  # realtime, so wait a bit for the next tick/frame

            # do all the functional stuff here
            self.world.tick(self.t[counter])

            # time keeping
            t_elapsed = (pygame.time.get_ticks() - t0) * 1e-3

            # draw the world
            self.world.draw(self.window, self.ppm)

            # simulator state
            sim_state_text = ["t_real: {:.2f}".format(t_elapsed),
                              "t_sim: {:.2f}".format(self.t[counter])]

            pos = (5, self.window.get_height() - 5)
            font = pygame.font.SysFont("verdana", 12)
            for txt in sim_state_text:
                txt_surface = font.render(txt, True, (0, 0, 0))
                p = txt_surface.get_rect(left=pos[0], bottom=pos[1])
                self.window.blit(txt_surface, p)
                pos = (pos[0], p.top)

            pygame.display.flip()

            counter += 1

            if self.dt * counter >= self.T:
                running = False
                print("Time's up, we're done here. Simulation finished in {0} seconds".format(t_elapsed))

        # self.plot_stuff()
        # self.save_stuff()

    def save_stuff(self, filename='trajectory'):
        human_trajectory = self.world.agents["human"].trajectory
        av_trajectory = self.world.agents["av"].trajectory
        human_trajectory.data.to_csv("data/human_" + filename + ".csv")
        av_trajectory.data.to_csv("data/av_" + filename + ".csv")

    def plot_stuff(self):
        human_trajectory = self.world.agents["human"].trajectory
        av_trajectory = self.world.agents["av"].trajectory

        fig, axs = plt.subplots(4, 1)

        # velocity
        axs[0].plot(human_trajectory.t, human_trajectory.x[3, :], color=self.world.agents["human"].color)
        axs[0].plot(av_trajectory.t, av_trajectory.x[3, :], color=self.world.agents["av"].color)
        # axs[0].set_xlabel('t, s')
        axs[0].set_ylabel('$v$, m/s')
        axs[0].legend(['human', 'av'])

        # phi
        axs[1].plot(human_trajectory.t, human_trajectory.x[2, :], color=self.world.agents["human"].color)
        axs[1].plot(av_trajectory.t, av_trajectory.x[2, :], color=self.world.agents["av"].color)
        # axs[1].set_xlabel('t, s')
        axs[1].set_ylabel('$\psi$, rad')

        # acceleration / deceleration
        axs[2].plot(human_trajectory.t, human_trajectory.u[0, :] + human_trajectory.u[1, :], color=self.world.agents["human"].color)
        axs[2].plot(av_trajectory.t, av_trajectory.u[0, :] + av_trajectory.u[1, :], color=self.world.agents["av"].color)
        # axs[2].set_xlabel('t, s')
        axs[2].set_ylabel('$u_{acc}$, rad')

        # steering wheel
        axs[3].plot(human_trajectory.t, human_trajectory.u[2, :], color=self.world.agents["human"].color)
        axs[3].plot(av_trajectory.t, av_trajectory.u[2, :], color=self.world.agents["av"].color)
        axs[3].set_xlabel('t, s')
        axs[3].set_ylabel('$\delta_r$, rad')

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    # a whole new woooooorld
    my_world = IntersectionWorld(0.1, 80., 100.)

    # No one to tell us no
    # Or where to go
    # Or say we're only dreaming
    scenario1 = scenarios.scenario_demo_1(my_world)

    sim = Simulator(scenario1, T=15., dt=0.1)
    sim.run()
