import numpy as np
import ddm
import math
from scipy import interpolate


class HumanModel:
    # TODO: extend the models with time gap and acceleration
    # TODO: manage model parameters properly
    def __init__(self):
        pass

    def get_decision(self, distance_gap):
        pass

    def get_av_policy_cost(self, av_distance, av_velocity, av_acceleration):
        pass


class HumanModelDelayedThreshold(HumanModel):
    def __init__(self, critical_gap, noise_intensity, delay_mean, delay_std):
        self.critical_gap = critical_gap
        self.noise_intensity = noise_intensity
        self.random_delay = delay_mean + np.random.randn() * delay_std

    def get_decision(self, distance_gap, time_elapsed):
        if time_elapsed > self.random_delay:
            return "turn" if distance_gap > self.critical_gap + np.random.randn() * self.noise_intensity else "wait"
        else:
            return None


class HumanModelDDMStaticDrift(HumanModel):
    # TODO: this will be obsolete one the dynamic drift model is implemented
    def __init__(self, critical_gap, boundary, drift_rate, diffusion_rate, dt):
        self.evidence = 0
        self.critical_gap = critical_gap
        self.boundary = boundary
        self.drift_rate = drift_rate
        self.diffusion_rate = diffusion_rate
        self.dt = dt

    def get_decision(self, distance_gap, time_elapsed):
        print(self.evidence)
        self.evidence += (self.drift_rate * (distance_gap - self.critical_gap) * self.dt
                          + np.random.randn() * self.diffusion_rate) * np.sqrt(self.dt)
        if abs(self.evidence) > self.boundary:
            return "turn" if self.evidence > self.boundary else "wait"
        else:
            return None


class HumanModelDDMDynamicDrift(HumanModel):
    name = "Drift-diffusion model with the drift rate varying with oncoming vehicle's trajectory"
    param_names = ["drift_rate", "alpha_d", "alpha_time_gap", "alpha_dtime_gap_dt", "theta",
                   "boundary", "nondecision_time_loc", "nondecision_time_scale"]
    weight_p_turn = 0.5
    weight_mean_rt = 0.5
    T_dur = 5

    class TimeVaryingDrift(ddm.Drift):
        name = "Drift dynamically depends on distance to the oncoming vehicle"
        required_parameters = ["drift_rate", "alpha_time_gap", "alpha_dtime_gap_dt", "theta",
                               "distance_gap_interp", "time_gap_interp", "dtime_gap_dt_interp"]

        def get_drift(self, t, conditions, **kwargs):
            return self.drift_rate * (self.distance_gap_interp(t)
                                      + self.alpha_time_gap * self.time_gap_interp(t)
                                      + self.alpha_dtime_gap_dt * self.dtime_gap_dt_interp(t)
                                      - self.theta)

    def __init__(self, drift_rate=None, alpha_time_gap=None, alpha_dtime_gap_dt=None, theta=None, boundary=None,
                 nondecision_time_loc=None, nondecision_time_scale=None,
                 t=None, distance_gap=None, time_gap=None, dtime_gap_dt=None):
        distance_gap_interp = interpolate.interp1d(t, distance_gap)
        time_gap_interp = interpolate.interp1d(t, time_gap)
        dtime_gap_dt_interp = interpolate.interp1d(t, dtime_gap_dt)

        self.model = ddm.Model(
            # running time with time-varying drift: 10s :/
            drift=self.TimeVaryingDrift(drift_rate=drift_rate, alpha_time_gap=alpha_time_gap,
                                        alpha_dtime_gap_dt=alpha_dtime_gap_dt, theta=theta,
                                        distance_gap_interp=distance_gap_interp, time_gap_interp=time_gap_interp,
                                        dtime_gap_dt_interp=dtime_gap_dt_interp),
            # running time with constant drift: 0.25s :/
            # drift=ddm.DriftConstant(drift=0.2),
            noise=ddm.NoiseConstant(noise=1),
            bound=ddm.BoundConstant(B=boundary),
            # TODO: investigate why adding simple nondecision time overlay sometimes throws an exception
            # overlay=ddm.OverlayNonDecisionUniform(nondectime=nondecision_time_loc, halfwidth=nondecision_time_scale),
            T_dur=self.T_dur)

    def get_av_policy_cost(self):
        solution = self.model.solve()
        p_turn = solution.prob_correct()
        mean_rt_turn = solution.mean_decision_time()
        # FIXME: pyddm doesn't provide mean decision time for error (in our case, wait) decisions; submit a pull request
        mean_rt_wait = math.fsum(solution.err * solution.model.t_domain()) / solution.prob_correct()
        mean_rt = p_turn * mean_rt_turn + (1 - p_turn) * mean_rt_wait

        return -self.weight_p_turn * (p_turn - 0.5) ** 2 + self.weight_mean_rt * mean_rt


class HumanModelDDMDynamicDriftFittable:
    name = "Drift-diffusion model with the drift rate varying with distance, time gap, and acceleration of the " \
           "oncoming car "
    param_names = ["drift_rate", "alpha_d", "alpha_a", "theta",
                   "boundary", "nondecision_time_loc", "nondecision_time_scale"]
    T_dur = 3

    class TimeVaryingDrift(ddm.models.Drift):
        name = "Drift dynamically depends on distance to the oncoming vehicle"
        required_parameters = ["drift_rate", "alpha_d", "alpha_a", "theta"]
        required_conditions = ["d_condition", "tau_condition", "a_condition"]

        def get_drift(self, t, conditions, **kwargs):
            v = conditions['d_condition'] / conditions['tau_condition']
            drift = self.drift_rate * (conditions['tau_condition'] - t
                                       + self.alpha_d * (conditions['d_condition'] - v * t)
                                       + self.alpha_a * conditions["a_condition"]
                                       - self.theta)
            # print("params: " + str([self.drift_rate, self.alpha_d, self.alpha_a, self.theta]))
            # print("Drift rate: " + str(self.drift_rate))
            # print("tau drift: {0}".format(str(conditions['tau_condition'] - t)))
            # print("d drift: {0}".format(str(self.alpha_d * (conditions['d_condition'] - v * t))))
            # print("a drift: {0}".format(str(self.alpha_a * conditions["a_condition"])))
            # print("theta: " + str(self.theta))
            # print("Total drift: {0}".format(str(drift)))
            return drift

    def __init__(self):
        self.model = ddm.Model(
            drift=self.TimeVaryingDrift(drift_rate=ddm.Fittable(minval=0.1, maxval=3),
                                        alpha_d=ddm.Fittable(minval=0, maxval=1),
                                        alpha_a=ddm.Fittable(minval=-5, maxval=0),
                                        theta=ddm.Fittable(minval=4, maxval=40)),
            noise=ddm.NoiseConstant(noise=1),
            bound=ddm.BoundConstant(B=ddm.Fittable(minval=0.5, maxval=5)) ,
            # overlay=ddm.OverlayNonDecisionUniform(nondectime=ddm.Fittable(minval=0, maxval=1),
            #                                       halfwidth=ddm.Fittable(minval=0.001, maxval=0.4)),
            T_dur=self.T_dur, dt=.001)
