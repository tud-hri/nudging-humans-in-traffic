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


class DriftGeneralizedGap(pyddm.models.Drift):
    name = "Drift dynamically depends on the real-time values of distance, TTA, and acceleration"
    required_parameters = ["alpha", "beta_d", "beta_a", "theta"]
    required_conditions = ["d_condition", "tta_condition", "a_condition"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        return self.alpha * (self.beta_tta * (conditions["tta_condition"] - t)
                             + self.beta_d * (conditions["d_condition"] - t * conditions["d_condition"]
                                              / conditions["tta_condition"])
                             - self.beta_tta_or * (conditions["tta_or_condition"] - t)
                             - self.theta)


class BoundCollapsingTta(pyddm.models.Bound):
    # TODO: check if parameter r could be replaced with beta_tta_or
    name = "Bounds dynamically collapsing with TTA"
    required_parameters = ["b_0", "k", "r", "tta_crit"]
    required_conditions = ["tta_condition", "d_condition", "tta_or_condition"]

    def get_bound(self, t, conditions, **kwargs):
        tau = conditions["tta_condition"] - t
        tau_or = conditions["tta_or_condition"] - t
        return self.b_0 / (1 + np.exp(-self.k * (tau + self.r * tau_or - self.tta_crit)))


class ModelDynamicDriftCollapsingBounds:
    T_dur = 4.0
    param_names = ["alpha", "beta_d", "beta_tta_or", "theta", "b_0", "k", "r", "tta_crit",
                   "ndt_location", "ndt_scale"]

    def __init__(self):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))
        self.drift = DriftTtaDistance(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                      beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                      beta_tta_or=pyddm.Fittable(minval=0, maxval=1.0),
                                      theta=pyddm.Fittable(minval=0, maxval=20))

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.1, maxval=2.0),
                                        r=pyddm.Fittable(minval=0.0, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(
            name="Dynamic drift defined by real-time TTA and d, bounds collapsing with TTA and TTA_or",
            drift=self.drift, noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
            overlay=self.overlay, T_dur=self.T_dur)


class DriftGaze(DriftTtaDistance):
    name = "Drift dynamically depends on the real-time values of TTA and distance, modulated by gaze"
    required_parameters = ["alpha", "beta_d", "beta_tta_or", "theta", "gamma", "gaze_sample_f"]

    def get_drift(self, t, conditions, **kwargs):
        gaze_dependent_generalized_gap = ((self.beta_tta * (conditions["tta_condition"] - t)
                                           + self.beta_d * (conditions["d_condition"] - t * conditions["d_condition"]
                                                            / conditions["tta_condition"])) *
                                          max(self.gaze_sample_f(t), self.gamma)
                                          - self.beta_tta_or * (conditions["tta_or_condition"] - t) *
                                          max(1 - self.gaze_sample_f(t), self.gamma)
                                          - self.theta)

        return self.alpha * gaze_dependent_generalized_gap

class BoundCollapsingGeneralizedGap(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with the generalized gap"
    required_parameters = ["b_0", "k", "beta_d", "beta_tta_or", "theta", "gamma", "gaze_sample_f"]
    required_conditions = ["tta_condition", "d_condition", "tta_or_condition"]
    beta_tta = 1.0

    def get_bound(self, t, conditions, **kwargs):
        generalized_gap = ((self.beta_tta * (conditions["tta_condition"] - t)
                                           + self.beta_d * (conditions["d_condition"] - t * conditions["d_condition"]
                                                            / conditions["tta_condition"]))
                                          - self.beta_tta_or * (conditions["tta_or_condition"] - t)
                                          - self.theta)
        return self.b_0 / (1 + np.exp(-self.k * generalized_gap))

class BoundCollapsingGazeDependentGeneralizedGap(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with the generalized gap"
    required_parameters = ["b_0", "k", "beta_d", "beta_tta_or", "theta", "gamma", "gaze_sample_f"]
    required_conditions = ["tta_condition", "d_condition", "tta_or_condition"]
    beta_tta = 1.0

    def get_bound(self, t, conditions, **kwargs):
        gaze_dependent_generalized_gap = ((self.beta_tta * (conditions["tta_condition"] - t)
                                           + self.beta_d * (conditions["d_condition"] - t * conditions["d_condition"]
                                                            / conditions["tta_condition"])) *
                                          max(self.gaze_sample_f(t), self.gamma)
                                          - self.beta_tta_or * (conditions["tta_or_condition"] - t) *
                                          max(1 - self.gaze_sample_f(t), self.gamma)
                                          - self.theta)
        return self.b_0 / (1 + np.exp(-self.k * gaze_dependent_generalized_gap))

class ModelGazeDependent(ModelDynamicDriftCollapsingBounds):
    param_names = ["alpha", "beta_d", "beta_tta_or", "theta", "gamma", "b_0", "k", "r", "tta_crit",
                   "ndt_location", "ndt_scale"]

    def __init__(self, gaze_sample):
        super(ModelGazeDependent, self).__init__()
        t = np.linspace(0, self.T_dur, len(gaze_sample))
        gaze_sample_f = interpolate.interp1d(t, gaze_sample)

        # TODO: this assumes that gaze_sample is defined over T_dur - fix this
        self.drift = DriftGaze(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                               beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                               beta_tta_or=pyddm.Fittable(minval=0, maxval=1.0),
                               theta=pyddm.Fittable(minval=0, maxval=20),
                               gamma=pyddm.Fittable(minval=0, maxval=1.0),
                               gaze_sample_f=gaze_sample_f)

        self.model = pyddm.Model(name="Gaze-dependent drift, bounds collapsing with TTA and TTA_or",
                                 drift=self.drift, noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)


class ModelGazeDependentBoundGeneralizedGap(ModelDynamicDriftCollapsingBounds):
    param_names = ["alpha", "beta_d", "beta_tta_or", "theta", "gamma", "b_0", "k", "ndt_location", "ndt_scale"]

    def __init__(self, gaze_sample):
        super(ModelGazeDependentBoundGeneralizedGap, self).__init__()
        t = np.linspace(0, self.T_dur, len(gaze_sample))
        gaze_sample_f = interpolate.interp1d(t, gaze_sample)

        beta_d = pyddm.Fittable(minval=0.0, maxval=1.0)
        beta_tta_or = pyddm.Fittable(minval=0, maxval=1.0)
        theta = pyddm.Fittable(minval=0, maxval=20)
        gamma = pyddm.Fittable(minval=0, maxval=1.0)

        # this assumes that gaze_sample is defined over T_dur - fix this
        self.drift = DriftGaze(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                               beta_d=beta_d,
                               beta_tta_or=beta_tta_or,
                               theta=theta,
                               gamma=gamma,
                               gaze_sample_f=gaze_sample_f)

        self.bound = BoundCollapsingGeneralizedGap(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.1, maxval=2.0),
                                        beta_d=beta_d,
                                        beta_tta_or=beta_tta_or,
                                        theta=theta,
                                        gamma=gamma,
                                        gaze_sample_f=gaze_sample_f)

        self.model = pyddm.Model(name="Gaze-dependent drift, bounds collapsing with TTA and TTA_or",
                                 drift=self.drift, noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)
