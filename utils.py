"""
utils.py
Static functions for common utilities
"""

import numpy as np
import os
import csv
import pandas as pd
import scipy

def get_nudge_condition_map():
    return {"(0.0, 4, 4, 0.0)": "Long acceleration",
            "(0.0, 4, -4, 0.0)": "Acceleration nudge",
            "(0.0, 0.0, 0.0, 0.0)": "Constant speed",
            "(0.0, -4, 4, 0.0)": "Deceleration nudge",
            "(0.0, -4, -4, 0.0)": "Long deceleration"}


def rotmatrix(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def get_derivative(t, x):
    # To be able to reasonably calculate derivatives at the end-points of the trajectories,
    # append three extra points before and after the actual trajectory, so we get N+6
    # points instead of N
    # TODO: make a linear extrapolation instead of constant
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


def get_psf_ci(data):
    # psf: psychometric function
    # ci: dataframe with confidence intervals for probability per nudge level
    nudge_conditions = get_nudge_condition_map().values()

    psf = np.array([len(data[data.is_go_decision & (data.nudge_condition == nudge_condition)])
                    / len(data[data.nudge_condition == nudge_condition])
                    if len(data[(data.nudge_condition == nudge_condition)]) > 0 else np.NaN
                    for nudge_condition in nudge_conditions])

    ci = pd.DataFrame(psf, columns=["p_go"], index=nudge_conditions)

    n = [len(data[(data.nudge_condition == nudge_condition)]) for nudge_condition in nudge_conditions]
    ci["ci_l"] = ci["p_go"] - np.sqrt(psf * (1 - psf) / n)
    ci["ci_r"] = ci["p_go"] + np.sqrt(psf * (1 - psf) / n)

    return ci.reset_index().rename(columns={"index": "nudge_condition"})


def get_mean_sem(data, var="RT", groupby_var="nudge_condition", n_cutoff=2):
    mean = data.groupby(groupby_var)[var].mean()
    sem = data.groupby(groupby_var)[var].apply(lambda x: scipy.stats.sem(x, axis=None, ddof=0))
    n = data.groupby(groupby_var).size()
    data_mean_sem = pd.DataFrame({"mean": mean, "sem": sem, "n": n}, index=mean.index)
    data_mean_sem = data_mean_sem[data_mean_sem.n > n_cutoff]

    return data_mean_sem
