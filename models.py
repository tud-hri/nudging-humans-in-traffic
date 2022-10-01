import numpy as np
from scipy import stats
import pyddm
from scipy import interpolate

def get_state_interpolators(tta_0, d_0, a_values, a_duration):
    T_dur = 6.0
    breakpoints = np.array([0., 0.25, (0.25+a_duration), min(0.25 + a_duration*2, T_dur)] + [T_dur])

    # a_values = np.array([0.0, a_1, a_2, 0.0, 0.0])

    v_0 = d_0 / tta_0
    a_values = np.concatenate([a_values, [0.]])
    v_values = np.concatenate([[v_0], v_0 + np.cumsum(np.diff(breakpoints) * a_values[:-1])])
    d_values = np.concatenate([[d_0], d_0 - np.cumsum(np.diff(breakpoints) * (v_values[1:] + v_values[:-1]) / 2)])

    tta_values = d_values / v_values
    # if at some point the oncoming vehicle starts moving away from the intersection, tta goes negative
    # to avoid this, we create a bound on TTA: if v becomes small enough, TTA = tta_bound
    v_threshold = 1
    tta_values[v_values<v_threshold] = d_values[v_values<v_threshold] / v_threshold

    # acceleration is piecewise-constant
    f_a = interpolate.interp1d(breakpoints, a_values, kind=0)
    # under piecewise-constant acceleration, tta is piecewise-linear
    # f_v = interpolate.interp1d(acceleration_timings, v_condition, kind=1, fill_value=(v_0, v_0), bounds_error=False)
    f_tta = interpolate.interp1d(breakpoints, tta_values, kind=1)
    # under piecewise-linear v, d is piecewise-quadratic, but piecewise-linear approximation is very close
    f_d = interpolate.interp1d(breakpoints, d_values, kind=1)

    return f_tta, f_d, f_a

def f_get_env_state(t, conditions, a_duration=1):
    # f_tta, f_d, f_a = get_state_interpolators(conditions["tta_0"], conditions["d_0"], conditions["a_1"], conditions["a_2"])
    f_tta, f_d, f_a = get_state_interpolators(conditions["tta_0"], conditions["d_0"], conditions["a_values"], a_duration)
    tta = f_tta(t)
    d = f_d(t)
    a = f_a(t)
    return tta, d, a

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
    required_conditions = ["tta_0", "d_0"]

    def get_bound(self, t, conditions, **kwargs):
        tta = conditions["tta_0"] - t
        return self.b_0 / (1 + np.exp(-self.k * (tta - self.tta_crit)))


class DriftTtaDistance(pyddm.models.Drift):
    name = "Drift dynamically depends on the real-time values of distance, TTA, and acceleration"
    required_parameters = ["alpha", "beta_d", "theta"]
    required_conditions = ["tta_0", "d_0", "a_values"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        return self.alpha * (self.beta_tta * (conditions["tta_0"] - t)
                             + self.beta_d * conditions["d_0"] * (1 - t / conditions["tta_0"])
                             - self.theta)

class ModelTtaDistance:
    T_dur = 6.0
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
    required_parameters = ["alpha", "beta_d", "theta"]#, "f_get_env_state"]
    required_conditions = ["tta_0", "d_0"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        tta, d, a = f_get_env_state(t, conditions)
        return self.alpha * (self.beta_tta * tta + self.beta_d * d - self.theta)


class ModelFixedAcceleration:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "tta_crit", "ndt_location", "ndt_scale"]

    def __init__(self):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftFixedAcceleration(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                            beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                            theta=pyddm.Fittable(minval=0, maxval=20))#,
                                            # f_get_env_state=f_get_env_state)

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.1, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(name="Model 2", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)


class DriftAccelerationDependent(pyddm.models.Drift):
    name = "Drift depends on tta(t), d(t), and a(t)"
    required_parameters = ["alpha", "beta_d", "beta_a", "theta"]#, "f_get_env_state"]
    required_conditions = ["tta_0", "d_0", "a_values", "a_duration"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        tta, d, a = f_get_env_state(t, conditions, conditions["a_duration"])
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
                                            theta=pyddm.Fittable(minval=0, maxval=20))#,
                                            # f_get_env_state=f_get_env_state)

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.01, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(name="Model 3", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)

class DriftAccelerationDependent_v2(pyddm.models.Drift):
    name = "Drift depends on tta(t), d(t), and a(t)"
    required_parameters = ["alpha", "beta_d", "beta_a", "theta"]#, "f_get_env_state"]
    required_conditions = ["tta_0", "d_0", "a_1", "a_2"]
    # coefficient in front of tta is always 1.0
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        tta, d, a = f_get_env_state(t, conditions)
        return self.alpha * (self.beta_tta * tta + self.beta_d * d - self.beta_a*a - self.theta)


class ModelAccelerationDependent_v2:
    T_dur = 6.0
    param_names = ["alpha", "beta_d", "beta_a", "theta", "b_0", "k", "tta_crit", "ndt_location", "ndt_scale"]

    def __init__(self):
        self.overlay = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftAccelerationDependent_v2(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                            beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                            beta_a=pyddm.Fittable(minval=0.0, maxval=10.0),
                                            theta=pyddm.Fittable(minval=0, maxval=20))#,
                                            # f_get_env_state=f_get_env_state)

        self.bound = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                        k=pyddm.Fittable(minval=0.01, maxval=2.0),
                                        tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0))

        self.model = pyddm.Model(name="Model 4", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, T_dur=self.T_dur)