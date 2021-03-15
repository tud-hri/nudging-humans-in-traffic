import pygame
from casadi import *
from pygame.locals import *

from dynamics import CarDynamics
from utils import coordinate_transform


class Car:
    def __init__(self, p0, phi0: float, v0: float = 0., dt: float = 0.1, color: str = 'red'):
        x0 = np.array([p0[0], p0[1], phi0, v0])  # initial condition
        self.dt = dt
        self.dynamics = CarDynamics(dt, x0=x0)
        # self.trajectory = None # work in progress - at some point we might want to store state and input in a separate object (and the histories)
        self.u = np.zeros((2, 1))  # [acceleration, steering]
        self.x = x0  # [x, y, phi, v]
        self.world = None
        self.car_width = 2.  # width of the car
        self.car_length = self.dynamics.length

        self.image = pygame.image.load("img/car-{0}.png".format(color))

    def set_input(self, accelerate: float = 0., steer: float = 0.):
        """
        :param accelerate:
        :param steer:
        :return:
        """
        self.u[0] = accelerate
        self.u[1] = steer

    def tick(self):
        """
        Perform one integration step of the dynamics
        This is an override of the Entity.tick function, to enable us to define our own dynamics (e.g., as a CasADi function)
        :param dt:
        :return: car state
        """
        x_next = self.dynamics.integrate(self.x, self.u)
        self.x = x_next.full()  # casadi DM to np.array

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


class CarUserControlled(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., dt: float = 0.1, color: str = 'blue'):
        super(CarUserControlled, self).__init__(p0, phi0, v0, dt, color)
        self.accelerate_int = 0.
        self.steer_int = 0.

    def set_input(self, accelerate=0., steer=0.):
        """
        Crappy implementation of a simple keyboard controller
        :param accelerate:
        :param steer:
        :return:
        """
        accelerate_sensitivity = 2  # [m/s2 / s]
        decelerate_sensitivity = 3
        steer_sensitivity = 1.5 * np.pi  # [rad/s]

        keys = pygame.key.get_pressed()

        if not keys[K_UP] or not keys[K_DOWN]:
            self.accelerate_int = 0.
        if keys[K_UP]:
            self.accelerate_int += accelerate_sensitivity * self.dt
        elif keys[K_DOWN]:
            self.accelerate_int -= decelerate_sensitivity * self.dt
        accelerate = min(max(self.accelerate_int, -4.), 2.)  # limit acceleration to [-4., 2.]

        if not keys[K_LEFT] or not keys[K_RIGHT]:
            self.steer_int = 0.
        if keys[K_LEFT]:
            self.steer_int += steer_sensitivity * self.dt
        elif keys[K_RIGHT]:
            self.steer_int -= steer_sensitivity * self.dt
        steer = min(max(self.steer_int, -np.pi), np.pi)  # limit steer to [-pi, pi]

        super().set_input(accelerate, steer)


class CarSimpleOptimizer(Car):
    def __init__(self, p0, phi0: float, v0: float = 0., dt: float = 0.1, color: str = 'yellow'):
        super(CarSimpleOptimizer, self).__init__(p0, phi0, v0, dt, color)
        self.th = 2.  # time horizon (2 seconds)
        self.Nh = round(self.th / self.dt)  # number of steps in time horizon

        # desired state
        self.x_target = np.array([[p0[0]],
                                  [0.],
                                  [-np.pi / 2.],
                                  [50. / 3.6]])

        # setup the optimizer through CasADi
        nx = self.x.shape[0]
        self.opti = casadi.Opti()
        self.x_opti = self.opti.variable(nx, self.Nh + 1)
        self.u_opti = self.opti.variable(2, self.Nh)
        self.p_opti = self.opti.parameter(nx, 1)

        # set objective
        Q = np.diag([.5, 1e-6, 0.05, .5])
        dx = self.x_target - self.x_opti
        self.opti.minimize(sumsqr(Q @ dx) + sumsqr(self.u_opti))

        for k in range(0, self.Nh):
            self.opti.subject_to(self.x_opti[:, k + 1] == self.dynamics.integrate(x=self.x_opti[:, k], u=self.u_opti[:, k]))

        self.opti.subject_to(self.opti.bounded(-10, self.u_opti[0], 10))
        self.opti.subject_to(self.opti.bounded(-np.pi / 2., self.u_opti[1], np.pi / 2.))
        self.opti.subject_to(self.opti.bounded(-10. / 3.6, self.x_opti[3,:], 50. / 3.6))
        self.opti.subject_to(self.x_opti[:, 0] == self.x)

        # setup solver
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000, 'print_level': 0}
        self.opti.solver('ipopt', p_opts, s_opts)

    def set_input(self, accelerate=0., steer=0.):
        # set current state of initial condition
        self.opti.set_value(self.p_opti, self.x)

        # solve the problem!
        sol = self.opti.solve()

        # select the first index for the control input
        accelerate = sol.value(self.u_opti)[0, 0]
        steer = sol.value(self.u_opti)[1, 0]

        # print(self.x[3]*3.6)

        super().set_input(accelerate, steer)
