"""
utils.py
Static functions for common utilities
"""

import numpy as np
import os
import csv
import pandas as pd
import scipy

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

    # acceleration is piecewise-constant
    f_a = scipy.interpolate.interp1d(breakpoints, a_values, kind=0)
    # under piecewise-constant acceleration, v and tta is piecewise-linear
    f_tta = scipy.interpolate.interp1d(breakpoints, tta_values, kind=1)
    # under piecewise-linear v, d is piecewise-quadratic, but piecewise-linear approximation is very close in our case
    f_d = scipy.interpolate.interp1d(breakpoints, d_values, kind=1)

    return f_tta, f_d, f_a

def get_nudge_condition_map():
    return {(0.0, 4, 4, 0.0): "Long acceleration",
            (0.0, 4, -4, 0.0): "Acceleration nudge",
            (0.0, 0.0, 0.0, 0.0): "Constant speed",
            (0.0, -4, 4, 0.0): "Deceleration nudge",
            (0.0, -4, -4, 0.0): "Long deceleration"}

def get_derivative(t, x):
    # To be able to reasonably calculate derivatives at the end-points of the trajectories,
    # append three extra points before and after the actual trajectory, so we get N+6
    # points instead of N
    x = np.append(x[0] * np.ones(3), np.append(x, x[-1] * np.ones(3)))

    # Time vector is also artificially extended by equally spaced points
    # Use median timestep to add dummy points to the time vector
    timestep = np.median(np.diff(t))
    t = np.append(t[0] - np.arange(1, 4) * timestep, np.append(t, t[-1] + np.arange(1, 4) * timestep))

    # smooth noise-robust differentiators, see:
    # http://www.holoborodko.com/pavel/numerical-methods/ \
    # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
    v = (1 * (x[6:] - x[:-6]) / ((t[6:] - t[:-6]) / 6) +
         4 * (x[5:-1] - x[1:-5]) / ((t[5:-1] - t[1:-5]) / 4) +
         5 * (x[4:-2] - x[2:-4]) / ((t[4:-2] - t[2:-4]) / 2)) / 32

    return v


def write_to_csv(directory, filename, array, write_mode="a"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), write_mode, newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(array)


def get_psf_ci(data, var="is_go_decision"):
    # psf: psychometric function
    # ci: dataframe with confidence intervals for probability per nudge level
    conditions = get_nudge_condition_map().values()

    psf = np.array([len(data[data[var] & (data.condition == condition)])
                    / len(data[data.condition == condition])
                    if len(data[(data.condition == condition)]) > 0 else np.NaN
                    for condition in conditions])

    ci = pd.DataFrame(psf, columns=["p"], index=conditions)

    n = [len(data[(data.condition == condition)]) for condition in conditions]
    ci["ci_l"] = ci["p"] - np.sqrt(psf * (1 - psf) / n)
    ci["ci_r"] = ci["p"] + np.sqrt(psf * (1 - psf) / n)

    return ci.reset_index().rename(columns={"index": "condition"})


def get_mean_sem(data, var="RT", groupby_var="condition", n_cutoff=2):
    mean = data.groupby(groupby_var)[var].mean()
    sem = data.groupby(groupby_var)[var].apply(lambda x: scipy.stats.sem(x, axis=None, ddof=0, nan_policy="omit"))
    n = data.groupby(groupby_var).size()
    data_mean_sem = pd.DataFrame({"mean": mean, "sem": sem, "n": n}, index=mean.index)
    data_mean_sem = data_mean_sem[data_mean_sem.n > n_cutoff]

    return data_mean_sem
