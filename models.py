import numpy as np
from scipy import stats
import pyddm

class OverlayNonDecisionGaussian(pyddm.Overlay):
    """ Courtesy of the pyddm cookbook """
    name = "Add a Gaussian-distributed non-decision time"
    required_parameters = ["ndt_location", "ndt_scale"]

    def apply(self, solution):
        # Extract components of the solution object for convenience
        corr = solution.choice_upper
        err = solution.choice_lower
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
    required_parameters = ["b_0", "k", "tta_crit", "state_interpolators"]
    required_conditions = ["tta_0", "d_0"]

    def get_bound(self, t, conditions, **kwargs):
        f_tta, f_d, f_a = self.state_interpolators[str(conditions)]
        tta = f_tta(t)
        return self.b_0 / (1 + np.exp(-self.k * (tta - self.tta_crit)))

class BoundCollapsingGeneralizedGap(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with the generalized gap"
    required_parameters = ["b_0", "k", "alpha", "beta_d", "beta_a", "theta"]
    required_conditions = ["tta_0", "d_0"]
    beta_tta = 1.0

    def get_bound(self, t, conditions, **kwargs):
        f_tta, f_d, f_a = self.state_interpolators[str(conditions)]
        tta = f_tta(t)
        d = f_d(t)
        a = f_a(t)
        return self.b_0 / (1 + np.exp(-self.k * (self.beta_tta * tta + self.beta_d * d - self.beta_a*a - self.theta)))

class DriftAccelerationDependent(pyddm.models.Drift):
    # TODO: model fitting could potentially be sped up if the interpolators are passed to the Drift object
    #  by pyddm as parameters instead of being created from scratch every time the drift needs to be calculated,
    #  however, this doesn't work so far
    name = "Drift depends on tta(t), d(t), and a(t)"
    required_parameters = ["alpha", "beta_d", "beta_a", "theta", "state_interpolators"]
    required_conditions = ["tta_0", "d_0", "a_values", "a_duration"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        f_tta, f_d, f_a = self.state_interpolators[str(conditions)]
        tta = f_tta(t)
        d = f_d(t)
        a = f_a(t)
        return self.alpha * (self.beta_tta * tta + self.beta_d * d - self.beta_a*a - self.theta)


class ModelAccelerationDependent:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "tta_crit", "ndt_location", "ndt_scale"]

    def __init__(self):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftAccelerationDependent(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                            beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                            beta_a=pyddm.Fittable(minval=0.0, maxval=10.0),
                                            theta=pyddm.Fittable(minval=0, maxval=20))

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.001, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(name="Model 1", choice_names=('Go', 'Stay'), drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)


class ModelAccelerationDependentWithBias:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "tta_crit", "x0", "ndt_location", "ndt_scale"]

    def __init__(self, state_interpolators):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftAccelerationDependent(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                            beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                            beta_a=pyddm.Fittable(minval=0.0, maxval=10.0),
                                            theta=pyddm.Fittable(minval=0, maxval=20),
                                            state_interpolators=state_interpolators)

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.001, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0),
                                                state_interpolators=state_interpolators)

        self.IC = pyddm.ICPointRatio(x0=pyddm.Fittable(minval=-1.0, maxval=1.0))

        self.model = pyddm.Model(name="Model 2", choice_names=("Go", "Stay"),
                                 drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1),
                                 bound=self.bound,
                                 overlay=self.overlay,
                                 IC=self.IC,
                                 T_dur=self.T_dur)

class ModelGeneralizedGapWithBias:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "x0", "ndt_location", "ndt_scale"]

    def __init__(self):
        alpha = pyddm.Fittable(minval=0.0, maxval=5.0)
        beta_d = pyddm.Fittable(minval=0.0, maxval=1.0)
        beta_a = pyddm.Fittable(minval=0.0, maxval=10.0)
        theta = pyddm.Fittable(minval=0, maxval=20)

        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftAccelerationDependent(alpha=alpha, beta_d=beta_d, beta_a=beta_a, theta=theta)

        self.bound = BoundCollapsingGeneralizedGap(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                                    k=pyddm.Fittable(minval=0.001, maxval=2.0),
                                                   alpha=alpha, beta_d=beta_d, beta_a=beta_a, theta=theta)

        self.IC = pyddm.ICPointRatio(x0=pyddm.Fittable(minval=-1.0, maxval=1.0))

        self.model = pyddm.Model(name="Model 3", choice_names=("Go", "Stay"),
                                 drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1),
                                 bound=self.bound,
                                 overlay=self.overlay,
                                 IC=self.IC,
                                 T_dur=self.T_dur)

class ModelAccelerationIndependentConstantBounds:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "theta", "B", "x0", "ndt_location", "ndt_scale"]

    def __init__(self, state_interpolators):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftAccelerationDependent(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                                beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                                beta_a=0,
                                                theta=pyddm.Fittable(minval=0, maxval=20),
                                                state_interpolators=state_interpolators)
        self.bound = pyddm.BoundConstant(B=pyddm.Fittable(minval=0.1, maxval=5.0))

        self.IC = pyddm.ICPointRatio(x0=pyddm.Fittable(minval=-1.0, maxval=1.0))

        self.model = pyddm.Model(name="Model 4", choice_names=("Go", "Stay"),
                                 drift=self.drift, noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)

class ModelAccelerationDependentConstantBounds:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "B", "x0", "ndt_location", "ndt_scale"]

    def __init__(self, state_interpolators):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftAccelerationDependent(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                                beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                                beta_a=pyddm.Fittable(minval=0.0, maxval=10.0),
                                                theta=pyddm.Fittable(minval=0, maxval=20),
                                                state_interpolators=state_interpolators)
        self.bound = pyddm.BoundConstant(B=pyddm.Fittable(minval=0.1, maxval=5.0))

        self.IC = pyddm.ICPointRatio(x0=pyddm.Fittable(minval=-1.0, maxval=1.0))

        self.model = pyddm.Model(name="Model 5", choice_names=("Go", "Stay"),
                                 drift=self.drift, noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)