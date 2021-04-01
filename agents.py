import pygame
from casadi import *
from pygame.locals import *

from dynamics import CarDynamics
from human_models import HumanModel
from trajectory import Trajectory
from utils import coordinate_transform


class Car:
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'red'):
        x0 = np.array([p0[0], p0[1], phi0, v0])  # initial condition
        self.world = world
        self.dt = world.dt
        self.dynamics = CarDynamics(self.dt, x0=x0)
        self.u = np.zeros((2, 1))  # [acceleration, steering]
        self.x = x0  # [x, y, phi, v]
        self.trajectory = Trajectory(x0, self.u)
        self.world = world
        self.car_width = 2.  # width of the car
        self.car_length = self.dynamics.length
        self.color = color

        self.image = pygame.image.load("img/car-{0}.png".format(self.color))

    def calculate_action(self, sim_time: float):
        pass

    def tick(self, sim_time: float):
        """
        Perform one integration step of the dynamics
        This is an override of the Entity.tick function, to enable us to define our own dynamics
        (e.g., as a CasADi function)
        :param dt:
        :return: car state
        """
        x_next = self.dynamics.integrate(self.x, self.u)
        self.x = x_next.full()  # casadi DM to np.array

        # add new state and input to trajectory
        self.trajectory.append(sim_time, self.x, self.u)

    def draw(self, window, ppm):
        # coordinate transform to graphics coordinate frame
        p = self.x[0:2] * ppm
        p = coordinate_transform(p)

        img = pygame.transform.scale(self.image, (int(self.car_length * ppm), int(self.car_width * ppm)))
        img = pygame.transform.rotate(img, np.rad2deg(self.x[2]))

        # calculate center position for drawing
        img_rect = img.get_rect()
        img_rect.center = (p[0, 0], p[1, 0])

        window.blit(img, img_rect)

    def __str__(self):
        return "state: {}".format(self.x.T)

    def feature_collision(self, x_eval, sdx=4., sdy=1.5, x_ego_sym=None):
        """
        Represent the car's collision feature as a 2D gaussian
        :param x_eval: state vector to calculate reward for
        :param sdx: SD in car length
        :param sdy: SD in car width
        :return: accumulated reward
        """
        if x_ego_sym is None:
            x_ego = self.x
        else:
            x_ego = x_ego_sym

        theta = x_ego[2]
        n = x_eval.shape[1]
        f = MX(1, n)
        for k in range(0, n):
            d = (x_eval[0, k] - x_ego[0], x_eval[1, k] - x_ego[1])

            dh = casadi.cos(theta) * d[0] - casadi.sin(theta) * d[1]
            dw = casadi.sin(theta) * d[0] + casadi.cos(theta) * d[1]

            f[0, k] = 1. / (sdx * sqrt(2. * np.pi)) * casadi.exp(-0.5 * dh ** 2 / sdx ** 2) * 1. / (sdy * sqrt(2. * np.pi)) * casadi.exp(
                -0.5 * dw ** 2 / sdy ** 2)
        return f

    def text_state_render(self):
        font = pygame.font.SysFont("verdana", 12)
        text = "x: {0: .1f}, y: {1: .1f}, psi: {2: .2f}, v:{3: .1f}; u_a: {4: .2f}, u_phi: {5: .2f}".format(self.x[0, 0], self.x[1, 0], self.x[2, 0],
                                                                                                            self.x[3, 0], self.u[0, 0], self.u[1, 0])  #
        return font.render(text, True, self.color)

    @property
    def position(self):
        return self.x[0:2]

    @property
    def velocity(self):
        return self.x[3]


class CarUserControlled(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'blue'):
        super(CarUserControlled, self).__init__(p0, phi0, v0, world, color)
        self.accelerate_int = 0.
        self.steer_int = 0.

    def calculate_action(self, sim_time: float):
        accelerate_sensitivity = 2.  # [m/s2 / s]
        decelerate_sensitivity = 3.
        steer_sensitivity = 1.5 * np.pi  # [rad/s]

        keys = pygame.key.get_pressed()

        if not keys[K_UP] or not keys[K_DOWN]:
            self.accelerate_int = 0.
        if keys[K_UP]:
            self.accelerate_int += accelerate_sensitivity * self.dt
        elif keys[K_DOWN]:
            self.accelerate_int -= decelerate_sensitivity * self.dt
        accelerate = min(max(self.accelerate_int, -20.), 20.)  # limit acceleration to [-4., 2.]

        if not keys[K_LEFT] or not keys[K_RIGHT]:
            self.steer_int = 0.
        if keys[K_LEFT]:
            self.steer_int += steer_sensitivity * self.dt
        elif keys[K_RIGHT]:
            self.steer_int -= steer_sensitivity * self.dt
        steer = min(max(self.steer_int, -np.pi), np.pi)  # limit steer to [-pi, pi]

        self.u[0] = accelerate
        self.u[1] = steer


class CarMPC(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'yellow'):
        super(CarMPC, self).__init__(p0, phi0, v0, world, color)
        self.th = 1.5  # time horizon (2 seconds)
        self.Nh = round(self.th / self.dt)  # number of steps in time horizon

        # setup the optimizer through CasADi
        self.nx = self.x.shape[0]
        self.nu = 2  # two inputs, accelerate and steer
        self.opti = casadi.Opti()  # Opti() facilitates the NLP problem definition and solver

        self.x_mpc = np.zeros((self.nx, 1))
        self.cost_function = None
        self.v_des = 30 / 3.6  # desired velocity
        self.theta = np.zeros((7, 1))  # objective function weights [velocity, lane_center, boundary, input]

        # create symbolic variables
        self.x_opti = self.opti.variable(self.nx, self.Nh + 1)
        self.u_opti = self.opti.variable(self.nu, self.Nh)
        self.p_opti_x0 = self.opti.parameter(self.nx, 1)

        self.obstacles = None
        self.p_opti_x_obstacles = None  # placeholder - self.opti.parameter value to communicate the obstacle states to the solver

        self.set_constraints()  # set optimization problem constraints

        # setup solver
        p_opts = {'expand': True, 'print_time': 0}  # print_time stops printing the solver timing
        s_opts = {'max_iter': 1e6, 'print_level': 0}
        self.opti.solver('ipopt', p_opts, s_opts)

    def set_constraints(self):
        for k in range(0, self.Nh):
            self.opti.subject_to(self.x_opti[:, k + 1] == self.dynamics.integrate(x=self.x_opti[:, k], u=self.u_opti[:, k]))

        self.opti.subject_to(self.opti.bounded(-15, self.u_opti[0, :], 10))
        self.opti.subject_to(self.opti.bounded(-0.25 * np.pi, self.u_opti[1, :], 0.25 * np.pi))
        self.opti.subject_to(self.opti.bounded(0. / 3.6, self.x_opti[3, :], 80. / 3.6))
        self.opti.subject_to(self.p_opti_x0 == self.x_opti[:, 0])

        # # dynamic obstacle avoidance, based on Bruno Brito's paper: https://ieeexplore.ieee.org/document/8768044
        # phi_obstacle = self.p_opti_obstacle[2]
        # rot_phi = MX(2, 2)
        # rot_phi[0, 0] = casadi.cos(phi_obstacle)
        # rot_phi[0, 1] = -casadi.sin(phi_obstacle)
        # rot_phi[1, 0] = casadi.sin(phi_obstacle)
        # rot_phi[1, 1] = casadi.cos(phi_obstacle)
        #
        # # Compute ellipse matrix
        # r_disc = self.car_length / 2.
        # ab = np.array([[1. / ((self.obstacle_major + r_disc) ** 2), 0],
        #                [0, 1. / ((self.obstacle_minor + r_disc) ** 2)]])
        #
        # for k in range(0, self.Nh, 1):
        #     dx = MX(2, 1)
        #     dx[0] = self.p_opti_obstacle[0] - self.x_opti[0, k]
        #     dx[1] = self.p_opti_obstacle[1] - self.x_opti[1, k]
        #     self.opti.subject_to(dx.T @ rot_phi.T @ ab @ rot_phi @ dx > 1.)

    def set_objective(self, theta, primary_lanes, all_lanes=None, road_shoulders=None, obstacles=None, heading=None):

        self.theta = np.asarray(theta)

        # desired velocity
        self.cost_function = self.theta[0] * sumsqr(self.x_opti[3, :] - self.v_des)

        # desired heading
        if heading is not None:
            self.cost_function += self.theta[1] * sumsqr(self.x_opti[2, :] - heading)

        # add primary lane features
        for lane in primary_lanes:
            self.cost_function += self.theta[2] * sum2(lane.feature_lane_center(c=0.2, x=self.x_opti))

        # add lane features
        if all_lanes is not None:
            for lane in all_lanes:
                self.cost_function += self.theta[3] * sum2(lane.feature_lane_center(c=0.2, x=self.x_opti))

        # add shoulder features
        if road_shoulders is not None:
            for shoulder in road_shoulders:
                self.cost_function += self.theta[4] * sum2(shoulder.feature_shoulder(c=2., x=self.x_opti))

        # add collision object features
        #fixme: needs better explanations
        # 1. store the obstacle list, 2. create CasADi optimization parameters that are used to update the obstacle's state in the opti problem, 3. create the objective function in CasADi symbolics.
        self.obstacles = obstacles
        if obstacles is not None:
            self.p_opti_x_obstacles = self.opti.parameter(self.nx, len(obstacles))  # assume obstacles have the same state [x,y,psi,v]
            for i in range(len(obstacles)):
                self.cost_function += self.theta[5] * sum2(
                    obstacles[i].feature_collision(sdx=4, sdy=1.5, x_eval=self.x_opti, x_ego_sym=self.p_opti_x_obstacles[:,i]))

        # input / control effort
        self.cost_function += self.theta[6] * sumsqr(self.u_opti)

        self.opti.minimize(self.cost_function)

    def solve_opt_problem(self):
        u = np.zeros((self.nu, 1))

        # try:
        self.opti.set_value(self.p_opti_x0, self.x)  # set current state of initial condition

        # update the obstacle positions in the
        if self.obstacles is not None:
            for i in range(len(self.obstacles)):
                self.opti.set_value(self.p_opti_x_obstacles[:, i], self.obstacles[i].x)

        sol = self.opti.solve()  # solve the problem!

        # select the first index for the control input
        u[0] = sol.value(self.u_opti)[0, 0]
        u[1] = sol.value(self.u_opti)[1, 0]
        self.x_mpc = sol.value(self.x_opti)

        # except Exception as e:
        #     # no solution found, we can use this to add a breakpoint to use Casadi's debugger here.
        #     print(e)

        return u

    def calculate_action(self, sim_time: float):
        self.u = self.solve_opt_problem()

    def draw(self, window, ppm):
        super().draw(window, ppm)

        # show planned path (convert from m to pixels, and then coordinate transform)
        p = self.x_mpc[0:2, :] * ppm
        if p.shape[1] > 1:
            pygame.draw.lines(window, self.color, False, [tuple(coordinate_transform(x)) for x in p.T.tolist()])


class CarSimulatedHuman(CarMPC):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, human_model: HumanModel = None, color: str = 'red'):
        super(CarSimulatedHuman, self).__init__(p0, phi0, v0, world, color)
        self.human_model = human_model
        self.turning_time = 3.0  # how long the steer and acceleration commands are applied for after the decision is made

        self.decision = None
        self.t_decision = None
        self.is_turn_completed = False

    def calculate_action(self, sim_time: float):
        # fixme: this currently assumes that the human starts deciding from the very beginning of the simulation, might not be the case!
        if self.decision is None:
            print("The simulated human is thinking...")
            centers = [agent.position for agent in self.world.agents.values()]
            distance_gap = np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2)
            self.decision = self.human_model.get_decision(distance_gap, sim_time)
            if (self.decision == "turn") | (self.decision == "wait"):
                print("The simulated human has decided to %s" % self.decision)
                self.t_decision = sim_time
                print("Response time %.2fs" % self.t_decision)

        if self.decision == "turn":
            super().calculate_action(sim_time)  # let MPC set the input for the car
            if self.x[0] < 20.:
                self.is_turn_completed = True

            # super().set_input(*((self.steer, self.acceleration) if not self.is_turn_completed else (0, 0)))
            # if self.time_elapsed > self.t_decision + self.turning_time:
            #     self.is_turn_completed = True
