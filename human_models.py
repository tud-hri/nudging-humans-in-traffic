import numpy as np
import ddm


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

        self.pyddm_model = self.TimeVaryingDriftDDM()


class HumanModelDDMStaticDrift(HumanModel):
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
    param_names = ["critical_gap", "boundary", "drift_rate", "diffusion_rate",
                   "nondecision_time_loc", "nondecision_time_scale"]
    # TODO: get rid of the hardcoded prediction horizon
    T_dur = 5

    class TimeVaryingDrift(ddm.Drift):
        name = "Drift dynamically depends on distance, velocity, and acceleration of the oncoming vehicle"
        required_parameters = ["critical_gap", "boundary", "drift_rate", "diffusion_rate",
                               "nondecision_time_loc", "nondecision_time_scale"]
        required_conditions = ["x", "v", "a"]

        def get_drift(self, t, conditions, **kwargs):
            #TODO: implement the trajectory-dependent drift using the below code as an example
            # # not sure if passing x, v, and a arrays as conditions would work but worth trying
            # v = conditions['d_condition'] / conditions['tta_condition']
            # return (self.alpha * (conditions['tta_condition'] - t
            #                       + self.beta * (conditions['d_condition'] - v * t) - self.theta))
            pass

    def __init__(self):
        self.model = ddm.Model(
            drift=self.TimeVaryingDrift(),
            noise=ddm.NoiseConstant(noise=1),
            bound=ddm.BoundConstant(B=self.boundary),
            overlay=ddm.OverlayNonDecisionUniform(nondectime=self.nondecision_time_loc,
                                                  halfwidth=self.nondecision_time_scale),
            T_dur=self.T_dur)

    def get_decision(self, distance_gap, time_elapsed):
        # TODO: simulate this using self.model.simulate_trial
        pass

    def get_av_policy_cost(self, av_distance, av_velocity, av_acceleration):
        pass
