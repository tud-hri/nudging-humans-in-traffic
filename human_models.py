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


class HumanModelDDMDynamicDrift(HumanModel, ddm.Model):
    name = "Drift-diffusion model with the drift rate varying with oncoming vehicle's trajectory"
    param_names = ["drift_rate", "alpha_d", "alpha_time_gap", "alpha_dtime_gap_dt", "theta", "boundary",
                   "nondecision_time_loc", "nondecision_time_scale"]
    # TODO: get rid of the hardcoded prediction horizon
    T_dur = 5

    class TimeVaryingDrift(ddm.Drift):
        name = "Drift dynamically depends on distance, velocity, and acceleration of the oncoming vehicle"
        required_parameters = ["drift_rate", "alpha_d", "alpha_time_gap", "alpha_dtime_gap_dt", "theta"]
        required_conditions = ["time", "distance_gap", "time_gap", "dtime_gap_dt"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get_drift(self, t, conditions, **kwargs):
            # TODO: passing arrays as conditions doesn't work; to fix, either bypass pyddm's checks or pass them as
            #  extra arguments to the constructor (not sure how pyddm would react to that though...)
            distance_gap_interp = interpolate.interp1d(conditions["time"], conditions["distance_gap"])
            distance_gap_t = distance_gap_interp(t)
            return self.drift_rate * (distance_gap_t - self.theta)

    def __init__(self):
        self.model = ddm.Model(
            drift=self.TimeVaryingDrift(drift_rate=1, alpha_d=1, alpha_time_gap=1, alpha_dtime_gap_dt=1, theta=1),
            noise=ddm.NoiseConstant(noise=1),
            bound=ddm.BoundConstant(B=self.boundary),
            overlay=ddm.OverlayNonDecisionUniform(nondectime=self.nondecision_time_loc,
                                                  halfwidth=self.nondecision_time_scale),
            T_dur=self.T_dur)

    def get_decision(self, distance_gap, time_elapsed):
        # TODO: simulate this using self.model.simulate_trial
        pass

    def get_av_policy_cost(self, distance_to_av, rel_velocity, rel_acceleration):
        weight_p_turn = 0.5
        weight_mean_rt = 0.5
        solution = self.model.solve()
        p_turn = solution.prob_correct()
        mean_rt_turn = solution.mean_decision_time()
        # FIXME: pyddm doesn't provide mean decision time for error (in our case, wait) decisions; submit a pull request
        mean_rt_wait = math.fsum(solution.err * solution.model.t_domain()) / solution.prob_correct()
        mean_rt = p_turn * mean_rt_turn + (1 - p_turn) * mean_rt_wait

        return weight_p_turn * (p_turn - 0.5) ** 2 - weight_mean_rt * mean_rt
