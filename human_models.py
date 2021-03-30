import numpy as np


class HumanModel:
    # TODO: extend the models with time gap and acceleration
    # TODO: manage model parameters properly
    def __init__(self):
        pass

    def get_decision(self, distance_gap):
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


class HumanModelEvidenceAccumulation(HumanModel):
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
