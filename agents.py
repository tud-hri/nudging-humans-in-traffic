import numpy as np
from human_models import HumanModel

import carlo.agents
from carlo.entities import Point
from carlo.interactive_controllers import KeyboardController
from dynamics import CarDynamics


class Car(carlo.agents.Car):
    def __init__(self, world, center: Point, heading: float, dt: float, color: str = 'red'):
        super(Car, self).__init__(center, heading, color)
        x0 = np.array([center.x, center.y, heading, 0.])  # initial condition
        self.dynamics = CarDynamics(dt, x0=x0)
        # work in progress:at some point we might want to store state and input in a separate object (and the histories)
        # self.trajectory = None
        self.u_input = np.zeros((2, 1))  # [acceleration, steering]
        self.x_state = x0  # [x, y, phi, v]
        self.world = world

    def set_control(self, inputSteering: float, inputAcceleration: float):
        """
        Override from CARLO: set self.u_input
        :param inputSteering:
        :param inputAcceleration:
        :return:
        """
        self.u_input[0] = inputAcceleration
        self.u_input[1] = inputSteering

    def tick(self, dt: float):
        """
        Perform one integration step of the dynamics
        This is an override of the Entity.tick function, to enable us to define our own dynamics
        (e.g., as a CasADi function)
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
    def __init__(self, world, center: Point, heading: float, control_input, dt: float, color: str = 'red'):
        super(CarHardCoded, self).__init__(world, center, heading, dt, color)
        self.u = control_input
        self.k = 0  # index / time step

    def set_control(self):
        steer = self.u[0, self.k]
        accelerate = self.u[1, self.k]

        super().set_control(steer, accelerate)

        self.k += 1


class CarUserControlled(Car):
    def __init__(self, world, center: Point, heading: float, dt: float, color: str = 'blue'):
        super(CarUserControlled, self).__init__(world, center, heading, dt, color)
        self.controller = None

    def set_control(self):
        if self.controller is None:
            self.controller = KeyboardController(self.world)

        steer = min(max(self.controller.steering, -np.pi), np.pi)  # limit steer to [-pi, pi]
        accelerate = min(max(self.controller.throttle, -4.), 2.)  # limit acceleration to [-4., 2.]

        super().set_control(steer, accelerate)


class CarSimulatedHuman(Car):
    def __init__(self, world, human_model: HumanModel, center: Point, heading: float, dt: float, color: str):
        super(CarSimulatedHuman, self).__init__(world, center, heading, dt, color)
        self.human_model = human_model
        self.steer = 0.43  # in radian
        self.acceleration = 3  # in m/s^2
        self.turning_time = 3.0  # how long the steer and acceleration commands are applied for after the decision is made
        self.dt = dt
        # fixme: might not be the best idea to keep track of time inside the Car object
        self.time_elapsed = 0
        self.k = 0  # index / time step

        self.decision = None
        self.t_decision = None
        self.is_turn_completed = False

    def set_control(self):
        # fixme: this currently assumes that the human starts deciding from the very beginning of the simulation, might not be the case!
        if self.decision is None:
            print("The simulated human is thinking...")
            centers = [agent.center for agent in self.world.dynamic_agents]
            distance_gap = np.sqrt((centers[0].x - centers[1].x)**2 + (centers[0].y - centers[1].y)**2)
            self.decision = self.human_model.get_decision(distance_gap, self.time_elapsed)
            if (self.decision == "turn") | (self.decision == "wait"):
                print("The simulated human has decided to %s" % self.decision)
                self.t_decision = self.time_elapsed
                print("Response time %.2fs" % self.t_decision)

        if self.decision == "turn":
            super().set_control(*((self.steer, self.acceleration) if not self.is_turn_completed else (0, 0)))
            if self.time_elapsed > self.t_decision + self.turning_time:
                self.is_turn_completed = True

        # fixme: also might not be a good way of keeping track of time...
        self.time_elapsed += self.dt
