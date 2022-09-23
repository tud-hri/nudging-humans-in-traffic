import numpy as np
from scipy import stats
import pyddm
from scipy import interpolate


class OverlayNonDecisionGaussian(pyddm.Overlay):
    """ Courtesy of the pyddm cookbook """
    name = "Add a Gaussian-distributed non-decision time"
    required_parameters = ["ndt_location", "ndt_scale"]

    def apply(self, solution):
        # Extract components of the solution object for convenience
        corr = solution.corr
        err = solution.err
        dt = solution.model.dt
        # Create the weights for different timepoints
        times = np.asarray(list(range(-len(corr), len(corr)))) * dt
        weights = stats.norm(scale=self.ndt_scale, loc=self.ndt_location).pdf(times)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)  # Ensure it integrates to 1
        newcorr = np.convolve(weights, corr, mode="full")[len(corr):(2 * len(corr))]
        newerr = np.convolve(weights, err, mode="full")[len(corr):(2 * len(corr))]
        return pyddm.Solution(newcorr, newerr, solution.model,
                              solution.conditions, solution.undec)


class BoundCollapsingTta(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with TTA"
    required_parameters = ["b_0", "k", "tta_crit"]
    required_conditions = ["tta_0", "d_0", "a_condition"]

    def get_bound(self, t, conditions, **kwargs):
        tta = conditions["tta_condition"] - t
        return self.b_0 / (1 + np.exp(-self.k * (tta - self.tta_crit)))


class DriftTtaDistance(pyddm.models.Drift):
    name = "Drift dynamically depends on the real-time values of distance, TTA, and acceleration"
    required_parameters = ["alpha", "beta_d", "theta"]
    required_conditions = ["d_0", "tta_0"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        return self.alpha * (self.beta_tta * (conditions["tta_condition"] - t)
                             + self.beta_d * conditions["d_condition"] * (1 - t / conditions["tta_condition"])
                             - self.theta)

class ModelTtaDistance:
    T_dur = 5.0
    param_names = ["alpha", "beta_d", "theta", "b_0", "k", "tta_crit", "ndt_location", "ndt_scale"]

    def __init__(self):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0.0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))
        self.drift = DriftTtaDistance(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                      beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                      theta=pyddm.Fittable(minval=0, maxval=20))
        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.1, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(name="Model 1", drift=self.drift, noise=pyddm.NoiseConstant(noise=1),
                                 bound=self.bound, overlay=self.overlay, T_dur=self.T_dur)

class DriftFixedAcceleration(pyddm.models.Drift):
    name = "Same as DriftTtaDistance, but TTA(t) and d(t) are calculated through a function supplied externally"
    required_parameters = ["alpha", "beta_d", "theta", "f_get_env_state"]
    required_conditions = ["d_0", "tta_0", "a_condition"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        tta, d, a = self.f_get_env_state(t, conditions)
        return self.alpha * (self.beta_tta * tta + self.beta_d * d - self.theta)


class ModelFixedAcceleration:
    T_dur = 4.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "tta_crit", "ndt_location", "ndt_scale"]

    def __init__(self, f_get_env_state):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftFixedAcceleration(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                            beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                            theta=pyddm.Fittable(minval=0, maxval=20),
                                            f_get_env_state=f_get_env_state)

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.1, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(name="Model 2", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)


class ModelAccelerationDependent():
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "tta_crit", "ndt_location", "ndt_scale"]

    def __init__(self, gaze_sample):
        super(ModelAccelerationDependent, self).__init__()
        t = np.linspace(0, self.T_dur, len(gaze_sample))

        get_env_state_f = interpolate.interp1d(t, gaze_sample)

        # TODO: this assumes that gaze_sample is defined over T_dur - fix this
        self.drift = DriftGaze(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                               beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                               beta_tta_or=pyddm.Fittable(minval=0, maxval=1.0),
                               theta=pyddm.Fittable(minval=0, maxval=20),
                               gamma=pyddm.Fittable(minval=0, maxval=1.0),
                               gaze_sample_f=get_env_state_f)

        self.model = pyddm.Model(name="Gaze-dependent drift, bounds collapsing with TTA and TTA_or",
                                 drift=self.drift, noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)
