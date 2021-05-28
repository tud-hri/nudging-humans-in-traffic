import human_models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import ddm.plot
from datetime import datetime


def plot_policy(t, distance_gap, time_gap, dtime_gap_dt, a):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(t, distance_gap)
    ax1.set_ylabel("distance gap, m", fontsize=12)

    ax2.plot(t, time_gap)
    ax2.set_ylabel("time gap, s", fontsize=12)

    ax3.plot(t, dtime_gap_dt)
    ax3.plot(t, a)
    ax3.set_ylabel("time gap derivative / a", fontsize=12)

    ax3.set_xlabel("time, s", fontsize=12)

    plt.tight_layout()
    plt.show()


def visualize_model(model):
    ddm.plot.plot_fit_diagnostics(model)
    plt.show()


def get_policy_cost(t, distance_gap, time_gap, dtime_gap_dt):
    human_model = human_models.HumanModelDDMDynamicDrift(drift_rate=0.1, alpha_time_gap=0.1, alpha_dtime_gap_dt=0.1,
                                                         theta=45, boundary=1, nondecision_time_loc=0.3,
                                                         nondecision_time_scale=0.1, t=t, distance_gap=distance_gap,
                                                         time_gap=time_gap, dtime_gap_dt=dtime_gap_dt)

    model_policy_cost = human_model.get_av_policy_cost()
    visualize_model(model=human_model.model)

    return model_policy_cost


def test_get_policy_cost():
    av_trajectory = pd.read_csv("data/av_trajectory.csv")
    human_trajectory = pd.read_csv("data/human_trajectory.csv")
    t = av_trajectory.t
    distance_gap = np.sqrt((av_trajectory.x - human_trajectory.x[0]) ** 2
                           + (av_trajectory.y - human_trajectory.y[0]) ** 2)
    time_gap = distance_gap / av_trajectory.speed
    dtime_gap_dt = utils.get_derivative(t.to_numpy(), time_gap.to_numpy())
    a = utils.get_derivative(t.to_numpy(), av_trajectory.speed.to_numpy())

    start_time = datetime.now()
    policy_cost = get_policy_cost(t, distance_gap, time_gap, dtime_gap_dt)
    print("Model running time: %s" % str(datetime.now() - start_time))
    print("Policy cost: %.2f" % policy_cost)

    plot_policy(t, distance_gap, time_gap, dtime_gap_dt, a)


test_get_policy_cost()
