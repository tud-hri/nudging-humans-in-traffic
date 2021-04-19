import human_models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils


def plot_stuff(t, distance_gap, time_gap, dtime_gap_dt):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(t, distance_gap)
    ax1.set_ylabel("distance gap, m", fontsize=12)

    ax2.plot(t, time_gap)
    ax2.set_ylabel("time gap, s", fontsize=12)

    ax3.plot(t, dtime_gap_dt)
    ax3.set_ylabel("time gap derivative", fontsize=12)

    ax3.set_xlabel("time, s", fontsize=12)

    plt.tight_layout()
    plt.show()


def test_get_policy_cost():
    av_trajectory = pd.read_csv("data/av_trajectory.csv")
    human_trajectory = pd.read_csv("data/human_trajectory.csv")
    t = av_trajectory.t
    distance_gap = np.sqrt((av_trajectory.x - human_trajectory.x[0]) ** 2
                              + (av_trajectory.y - human_trajectory.y[0]) ** 2)
    time_gap = distance_gap / av_trajectory.speed
    dtime_gap_dt = utils.get_derivative(t.to_numpy(), time_gap.to_numpy())

    human_model = human_models.HumanModelDDMDynamicDrift()
    policy_cost = human_model.get_av_policy_cost(distance_gap, time_gap, dtime_gap_dt)

    print(policy_cost)

    plot_stuff(t, distance_gap, time_gap, dtime_gap_dt)

test_get_policy_cost()
