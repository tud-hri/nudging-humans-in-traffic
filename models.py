import numpy as np
import scipy.stats
import scipy.interpolate
import pyddm

def get_conditions():
    return([{"tta_0": tta_0, "d_0": d_0, "a_values": a_values, "a_duration": a_duration}
                  for tta_0 in [4.5, 5.5]
                  for d_0 in [80.0]
                  for a_values in [(0., 0., 0., 0.),
                                   (0., 4, 4, 0.),
                                   (0., 4, -4, 0.),
                                   (0., -4, 4, 0.),
                                   (0., -4, -4, 0.)]
                  for a_duration in [1.0]])

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
        weights = scipy.stats.norm(scale=self.ndt_scale, loc=self.ndt_location).pdf(times)
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

def get_state_interpolators(conditions):
    interpolators = [get_state_interpolators_per_condition(condition) for condition in conditions]
    return {str(condition): interpolator for condition, interpolator in zip(conditions, interpolators)}

def get_state_interpolators_per_condition(condition, T_dur=6.0):
    d_0 = condition["d_0"]
    tta_0 = condition["tta_0"]
    a_values = condition["a_values"]
    a_duration = condition["a_duration"]

    breakpoints = np.array([0., 0.25, (0.25+a_duration), min(0.25 + a_duration*2, T_dur)] + [T_dur])

    v_0 = d_0 / tta_0
    a_values = np.concatenate([a_values, [0.]])
    v_values = np.concatenate([[v_0], v_0 + np.cumsum(np.diff(breakpoints) * a_values[:-1])])
    d_values = np.concatenate([[d_0], d_0 - np.cumsum(np.diff(breakpoints) * (v_values[1:] + v_values[:-1]) / 2)])

    tta_values = d_values / v_values
    # if at some point the oncoming vehicle starts moving away from the intersection, tta goes negative
    # to avoid this, we create a bound on TTA: if v becomes small enough, TTA = tta_bound
    v_threshold = 1
    tta_values[v_values<v_threshold] = d_values[v_values<v_threshold] / v_threshold

    print(breakpoints)
    # acceleration is piecewise-constant
    f_a = scipy.interpolate.interp1d(breakpoints, a_values, kind=0)
    # under piecewise-constant acceleration, v and tta is piecewise-linear
    f_tta = scipy.interpolate.interp1d(breakpoints, tta_values, kind=1)
    # under piecewise-linear v, d is piecewise-quadratic, but piecewise-linear approximation is very close in our case
    f_d = scipy.interpolate.interp1d(breakpoints, d_values, kind=1)

    return f_tta, f_d, f_a

def get_model_components(state_interpolators):
    overlay_uniform = pyddm.OverlayNonDecisionUniform(nondectime=pyddm.Fittable(minval=0, maxval=2.0),
                                                         halfwidth=pyddm.Fittable(minval=0.001, maxval=0.5))
    overlay_gaussian = OverlayNonDecisionGaussian(ndt_location=pyddm.Fittable(minval=0, maxval=2.0),
                                                         ndt_scale=pyddm.Fittable(minval=0.001, maxval=0.5))

    drift_no_acceleration = DriftAccelerationDependent(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                                              beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                                              beta_a=0,
                                                              theta=pyddm.Fittable(minval=0, maxval=20),
                                                              state_interpolators=state_interpolators)

    drift_with_acceleration = DriftAccelerationDependent(alpha=pyddm.Fittable(minval=0.0, maxval=5.0),
                                                                beta_d=pyddm.Fittable(minval=0.0, maxval=1.0),
                                                                beta_a=pyddm.Fittable(minval=0.0, maxval=10.0),
                                                                theta=pyddm.Fittable(minval=0, maxval=20),
                                                                state_interpolators=state_interpolators)

    bound_constant = pyddm.BoundConstant(B=pyddm.Fittable(minval=0.1, maxval=5.0))

    bound_collapsing_tta = BoundCollapsingTta(b_0=pyddm.Fittable(minval=0.5, maxval=5.0),
                                                     k=pyddm.Fittable(minval=0.0, maxval=2.0),
                                                     tta_crit=pyddm.Fittable(minval=2.0, maxval=10.0),
                                                     state_interpolators=state_interpolators)

    IC_zero = pyddm.ICPointRatio(x0=0)

    IC_point_ratio = pyddm.ICPointRatio(x0=pyddm.Fittable(minval=-1.0, maxval=1.0))

    return overlay_gaussian, overlay_uniform, drift_no_acceleration, drift_with_acceleration, bound_constant, bound_collapsing_tta, IC_zero, IC_point_ratio

def get_model(model_no, T_dur):
    state_interpolators = get_state_interpolators(get_conditions())
    (overlay_gaussian, overlay_uniform,
     drift_no_acceleration, drift_with_acceleration,
     bound_constant, bound_collapsing_tta,
     IC_zero, IC_point_ratio) = get_model_components(state_interpolators=state_interpolators)

    overlay = overlay_gaussian

    if model_no == 1:
        drift = drift_no_acceleration
        bound = bound_constant
        IC = IC_zero
    elif model_no == 2:
        drift = drift_no_acceleration
        bound = bound_constant
        IC = IC_point_ratio
    elif model_no == 3:
        drift = drift_no_acceleration
        bound = bound_collapsing_tta
        IC = IC_zero
    elif model_no == 4:
        drift = drift_no_acceleration
        bound = bound_collapsing_tta
        IC = IC_point_ratio
    elif model_no == 5:
        drift = drift_with_acceleration
        bound = bound_constant
        IC = IC_zero
    elif model_no == 6:
        drift = drift_with_acceleration
        bound = bound_constant
        IC = IC_point_ratio
    elif model_no == 7:
        drift = drift_with_acceleration
        bound = bound_collapsing_tta
        IC = IC_zero
    elif model_no == 8:
        drift = drift_with_acceleration
        bound = bound_collapsing_tta
        IC = IC_point_ratio
    elif model_no == 9:
        # Model 2 but with uniform NDT
        drift = drift_no_acceleration
        bound = bound_constant
        IC = IC_point_ratio
        overlay = overlay_uniform

    return(pyddm.Model(name="Model %i" % model_no, choice_names=("Go", "Stay"),
                        drift=drift, bound=bound, IC=IC, overlay=overlay,
                        noise=pyddm.NoiseConstant(noise=1), T_dur=T_dur))

