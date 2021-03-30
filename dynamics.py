from casadi import *


# class Dynamics:
#     def __init__(self, nx: int, nu: int, f, dt: float):
#         self.dt = dt
#         self.f = f
#
#     def __call__(self, x, u):
#         return self.f(x, u)
#
#     def tick(self, dt):
#         pass


class CarDynamics:
    def __init__(self, dt: float, x0=None):
        """
        Create car dynamics using casadi
        :param dt: simulation time step
        :param x0: initial state
        """
        self.dt = dt

        # create casadi symbolic representations of the state and input vectors
        x = vertcat(MX.sym('x'), MX.sym('y'), MX.sym('phi'), MX.sym('v'))  # state vector: [x, y, phi, v]
        u = vertcat(MX.sym('a'), MX.sym('dr'))  # input vector: [a, dr]

        # simple bicycle dynamics, see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7225830
        self.length = 4. # assume a car is 4 meters; this should be a variable, really, but can't be bothered, hehe
        lr = self.length / 2.
        lf = lr
        beta = atan(lr / (lf + lr) * tan(u[1]))

        # ODE
        ode = vertcat(x[3] * cos(x[2] + beta),
                      x[3] * sin(x[2] + beta),
                      x[3] / lr * sin(beta),
                      u[0])

        f = Function('f', [x, u], [ode], ['x', 'u'], ['ode'])  # create a function for ease of calling

        # set up casadi integrator (runge-kutta, because why not)
        intg_opts = {'tf': dt,
                     'simplify': True,
                     'number_of_finite_elements': 4,  # number of intermediate integration steps
                     }

        # dae problem structure
        dae = {'x': x,  # states
               'p': u,  # parameters (fixed during the integration horizon
               'ode': f(x, u)}

        # create an integrator (runge kutta), evaluate it for one step, and retrieve the output
        intg = integrator('I', 'rk', dae, intg_opts)
        x_next = intg(x0=x, p=u)['xf']  # output is a struct, select final state xf

        # for ease of use, create a function that performs an integration step and returns the next state.
        # I have to say, casadi is quite particular in how it wants everything setup...
        self.F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    def integrate(self, x, u):
        """
        Call the casadi integration function we created
        :param x: np array, current state
        :param u: input vector
        :return: np array with the next state
        """
        return self.F(x=x, u=u)['x_next']

        pass
