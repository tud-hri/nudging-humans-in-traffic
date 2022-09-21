import pygame
from casadi import *
from pygame.locals import *

from dynamics import CarDynamics
from modeling_obsolete.human_models import HumanModel
from trajectory import Trajectory
from utils import coordinate_transform


class Car:
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'red'):
        x0 = np.array([[p0[0]], [p0[1]], [phi0], [v0]])  # initial condition [x, y, phi, v]
        self.world = world
        self.dt = world.dt
        self.dynamics = CarDynamics(self.dt, x0=x0)
        self.u = np.zeros((3, 1))  # [acceleration, deceleration, steering]
        self.x = x0  # [x, y, phi, v]
        self.trajectory = Trajectory(x0=x0, u0=self.u)
        self.world = world
        self.car_width = 2.  # width of the car
        self.car_length = self.dynamics.length
        self.color = color
        self.rect = pygame.Rect(round(p0[0] * 10), round(p0[1] * 10), 10, 10)

        self.image = pygame.image.load("img/car-{0}.png".format(self.color))

    def calculate_action(self, sim_time: float, step: int):
        pass

    def tick(self, sim_time: float, step: int):
        """
        Perform one integration step of the dynamics
        This is an override of the Entity.tick function, to enable us to define our own dynamics
        (e.g., as a CasADi function)
        :param sim_time: simulation time stamp
        :return: car state
        """

        x_next = self.dynamics.integrate(self.x, self.u)
        self.x = x_next.full()  # casadi DM to np.array

        # add new state and input to trajectory
        self.trajectory.append(sim_time, self.x, self.u)

    def draw(self, window, ppm):
        # coordinate transform to graphics coordinate frame
        p = coordinate_transform(self.x[0:2], ppm)

        img = pygame.transform.scale(self.image, (int(self.car_length * ppm), int(self.car_width * ppm)))
        img = pygame.transform.rotate(img, np.rad2deg(self.x[2]))

        # calculate center position for drawing
        self.rect = img.get_rect()
        self.rect.center = (p[0, 0], p[1, 0])

        window.blit(img, self.rect)

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

            f[0, k] = 1. / (sdx * sqrt(2. * np.pi)) * casadi.exp(-0.5 * dh ** 2 / sdx ** 2) * 1. / (
                    sdy * sqrt(2. * np.pi)) * casadi.exp(
                -0.5 * dw ** 2 / sdy ** 2)
        return f

    def text_state_render(self):
        font = pygame.font.SysFont("verdana", 12)
        text = "x: {0: .1f}, y: {1: .1f}, psi: {2: .2f}, v:{3: .1f} | u_a: {4:+.2f}, u_d: {5:-.2f}, u_dr: {6: .2f}".format(
            self.x[0, 0], self.x[1, 0],
            self.x[2, 0],
            self.x[3, 0], self.u[0, 0],
            self.u[1, 0], self.u[2, 0])  #
        return font.render(text, True, self.color)

    @property
    def position(self):
        return self.x[0:2]

    @property
    def velocity(self):
        return self.x[3]


class CarPredefinedControl(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'blue'):
        super(CarPredefinedControl, self).__init__(p0, phi0, v0, world, color)
        self.u_predefined = np.zeros(self.u.shape)

    def calculate_action(self, sim_time: float, step: int):
        # select the action from the predefined control trace

        # cap step within bounds of u_predefined
        step = min(max(0, step), self.u_predefined.shape[1] - 1)

        self.u = self.u_predefined[:, [step]]

        # constrain the car's velocity [0, 120km/h]
        # stop the car if stopped.
        if any(self.u[0:2] < 0) and self.x[3] <= 0.:
            self.u[0:2] = 0.

        if any(self.u[0:2] > 0) and self.x[3] >= 120./3.6:
            self.u[0:2] = 0.


class CarUserControlled(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'blue'):
        super(CarUserControlled, self).__init__(p0, phi0, v0, world, color)
        self.accelerate_int = 0.
        self.steer_int = 0.

    def calculate_action(self, sim_time: float, step: int):
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
        self.th = 2  # time horizon (2 seconds)
        self.Nh = round(self.th / self.dt)  # number of steps in time horizon

        # setup the optimizer through CasADi
        self.nx = self.x.shape[0]
        self.nu = 3  # three inputs, accelerate, decelerate, and steer
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

        # set optimization problem constraints
        self.set_constraints()

        # setup solver
        p_opts = {'expand': True, 'print_time': 0}  # print_time stops printing the solver timing
        s_opts = {'max_iter': 1e6, 'print_level': 0}
        self.opti.solver('ipopt', p_opts, s_opts)

    def set_constraints(self):
        for k in range(0, self.Nh):
            self.opti.subject_to(
                self.x_opti[:, k + 1] == self.dynamics.integrate(x=self.x_opti[:, k], u=self.u_opti[:, k]))

        self.opti.subject_to(self.opti.bounded(0., self.u_opti[0, :], 10.))  # acceleration, only positive, in m/s2
        self.opti.subject_to(self.opti.bounded(-20., self.u_opti[1, :], 0.))  # deceleration, only negative, in m/s2
        self.opti.subject_to(
            self.opti.bounded(-0.5 * np.pi, self.u_opti[2, :], 0.5 * np.pi))  # steering wheel input (rad)
        self.opti.subject_to(sumsqr(
            self.u_opti[0] * self.u_opti[1]) < 1e-6)  # product of acc / dec needs to be 0 (only acc or dec at a time)
        self.opti.subject_to(self.opti.bounded(0. / 3.6, self.x_opti[3, :], 80. / 3.6))  # speed
        self.opti.subject_to(self.p_opti_x0 == self.x_opti[:, 0])  # initial condition for each solver call

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
            # remove primary lanes so that they are not counted double in the cost function
            for lane in all_lanes:
                if lane not in primary_lanes:
                    self.cost_function += self.theta[3] * sum2(lane.feature_lane_center(c=0.2, x=self.x_opti))

        # add shoulder features
        if road_shoulders is not None:
            for shoulder in road_shoulders:
                self.cost_function += self.theta[4] * sum2(shoulder.feature_shoulder(c=3., x=self.x_opti))

        # add collision object features
        # 1. store the obstacle list,
        # 2. create CasADi optimization parameters that are used to update the obstacle's state in the opti problem,
        # 3. create the objective function in CasADi symbolics.
        self.obstacles = obstacles
        if obstacles is not None:
            self.p_opti_x_obstacles = self.opti.parameter(self.nx, len(
                obstacles))  # assume obstacles have the same state [x,y,psi,v]
            for i in range(len(obstacles)):
                self.cost_function += self.theta[5] * sum2(
                    obstacles[i].feature_collision(sdx=2.5, sdy=1.25, x_eval=self.x_opti,
                                                   x_ego_sym=self.p_opti_x_obstacles[:, i])
                )

        # input / control effort
        r = np.diag([1., 0., 2.5])  # relative weighting: considerably less weight on deceleration (encourage braking)
        self.cost_function += self.theta[6] * sumsqr(self.u_opti.T @ r @ self.u_opti)

        # set cost to minimize
        self.opti.minimize(self.cost_function)

    def solve_opt_problem(self):
        u = np.zeros((self.nu, 1))

        # try:
        self.opti.set_value(self.p_opti_x0, self.x)  # set current state of initial condition

        # update the obstacle positions in the
        if self.obstacles is not None:
            for i in range(len(self.obstacles)):
                self.opti.set_value(self.p_opti_x_obstacles[:, i], self.obstacles[i].x)

        # solve the problem!
        sol = self.opti.solve()

        # select the first index for the control input
        u[0] = sol.value(self.u_opti)[0, 0]
        u[1] = sol.value(self.u_opti)[1, 0]
        u[2] = sol.value(self.u_opti)[2, 0]
        self.x_mpc = sol.value(self.x_opti)

        # except Exception as e:
        #     # no solution found, we can use this to add a breakpoint to use Casadi's debugger here.
        #     print(e)

        return u

    def calculate_action(self, sim_time: float, step: int):
        self.u = self.solve_opt_problem()

    def draw(self, window, ppm):
        super().draw(window, ppm)

        # show planned path (convert from m to pixels, and then coordinate transform)
        p = self.x_mpc[0:2, :]
        if p.shape[1] > 1:
            pygame.draw.lines(window, self.color, False, [tuple(coordinate_transform(x, ppm)) for x in p.T.tolist()])


class CarSimulatedHuman(CarMPC):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, human_model: HumanModel = None, color: str = 'red'):
        super(CarSimulatedHuman, self).__init__(p0, phi0, v0, world, color)
        self.human_model = human_model
        self.turning_time = 3.0  # how long the steer and acceleration commands are applied for after the decision is made

        self.decision = None
        self.t_decision = None
        self.is_turn_completed = False

    def calculate_action(self, sim_time: float, step: int):
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


class CarHumanTriggeredPD(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., world=None, color: str = 'red'):
        super(CarHumanTriggeredPD, self).__init__(p0, phi0, v0, world, color)
        self.decision = None
        self.response_time = -1.
        # self.decision_go = False
        # self.decision_stay = False

        # desired state
        self.v_des = 50. / 3.6
        self.phi_des = np.pi
        self.y_des = world.lanes[3].p0[1]

        # gains (proportional only for now)
        self.K_v = 2.
        self.K_y = 0.23
        self.K_psi = 1.

    def calculate_action(self, sim_time: float, step: int):
        keys = pygame.key.get_pressed()

        # if go key
        if self.decision is None:
            if keys[K_LEFT] or keys[K_z]:
                self.decision = "go"
            elif keys[K_s] or keys[K_SLASH]:
                self.decision = "stay"

            if not (self.decision is None):
                self.response_time = sim_time
                # print(f"Decision: {self.decision}")
                # print(f"Response time: {self.response_time:.3f}")

        # if decision is made, use a simple PD to control the car
        if self.decision == "go":
            # super().calculate_action(sim_time)
            self.u = np.zeros((3, 1))

            # acceleration
            a = self.K_v * (self.v_des - self.x[3])
            a = min(max(a, -10), 5)

            if np.sign(a) == 1:
                self.u[0] = a
            elif np.sign(a) == -1:
                self.u[1] = a

            # steering
            s = self.K_psi * (self.phi_des - self.x[2]) - self.K_y * (self.y_des - self.x[1])
            self.u[2] = min(max(s, -np.pi), np.pi)
