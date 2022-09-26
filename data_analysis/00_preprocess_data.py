import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import random


def merge_csv_files(data_path):
    dfs = []
    raw_data_path = os.path.join(data_path, "raw")
    for file in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file)
        if file_path.endswith(".txt"):
            print(file_path)
            dfs.append(pd.read_csv(file_path, sep="\t"))
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join(data_path, "raw_data_merged.csv"), index=False, sep="\t")


def get_measures(traj):
    # print(traj.name)
    if (sum(traj.bot_v) > 0) & (sum(traj.throttle) > 0):
        # the bot started moving when the truck moved far enough not to block the participants" line of sight to the bot
        # so we can count on the first moment the bot velocity is non-zero as the moment the bot became visible first
        # this is what we take as the decision start time
        # idx_truck_moving = traj.truck_v.to_numpy().nonzero()[0][0]
        # idx_bot_visible = idx_bot_moving + np.argmax(traj.bot_angle[idx_bot_moving:] - traj.truck_angle[idx_bot_moving:] < 0)
        idx_bot_visible = traj.bot_v.to_numpy().nonzero()[0][0]
        throttle = traj.iloc[idx_bot_visible:, traj.columns.get_loc("throttle")]
        idx_gas_response = idx_bot_visible + (throttle > 0).to_numpy().nonzero()[0][0]
        RT_gas = traj.t.values[idx_gas_response] - traj.t.values[idx_bot_visible]
        idx_min_distance = idx_bot_visible + np.argmin(traj.d_ego_bot[idx_bot_visible:].values)
        min_distance = min(traj.d_ego_bot[idx_bot_visible:].values)
    else:
        idx_bot_visible = -1
        idx_gas_response = -1
        idx_min_distance = -1
        min_distance = -1
        RT_gas = -1

    let_pass = traj.iloc[idx_bot_visible:, traj.columns.get_loc("let_pass")]
    if sum(let_pass)>0:
        idx_yield = idx_bot_visible + (let_pass > 0).to_numpy().nonzero()[0][0]
        RT_yield = traj.t.values[idx_yield] - traj.t.values[idx_bot_visible]
    else:
        idx_yield = -1
        RT_yield = -1


    is_negative_rating = traj.subjective_bad.any()
    return pd.Series({"idx_bot_visible": idx_bot_visible,
                      "idx_response": idx_gas_response,
                      "idx_yield": idx_yield,
                      "idx_min_distance": idx_min_distance,
                      "min_distance": min_distance,
                      "RT_gas": RT_gas,
                      "RT_yield": RT_yield,
                      "is_negative_rating": is_negative_rating})

def process_data(data):
    data.loc[:,"t"] = data.t.groupby(data.index.names).transform(lambda t: (t-t.min()))

    data = data.rename(columns={"accl_profile_values": "a_condition"})

    condition_map = {"[0.0, 0.0, 0.0, 0.0]": "Constant speed",
                 "[0.0, -4, 4, 0.0]": "Deceleration nudge",
                 "[0.0, 4, -4, 0.0]": "Acceleration nudge",
                 "[0.0, -4, -4, 0.0]": "Deceleration",
                 "[0.0, 4, 4, 0.0]": "Acceleration"}
    data["nudge_condition"] = data["a_condition"].map(condition_map)

    # we are only interested in left turns
    data = data[data.turn_direction==1]

    # discarding the filler trials
    data = data[data.d_condition==80]

    # only consider the data recorded within 20 meters of each intersection
    data = data[abs(data.ego_distance_to_intersection)<20]

    # smooth the time series by filtering out the noise using Savitzky-Golay filter
    apply_filter = lambda traj: savgol_filter(traj, window_length=21, polyorder=2, axis=0)
    cols_to_smooth = ["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_ax", "ego_ay",
                      "bot_x", "bot_y", "bot_vx", "bot_vy", "bot_ax", "bot_ay"]
    data.loc[:, cols_to_smooth] = (data.loc[:, cols_to_smooth].groupby(data.index.names).transform(apply_filter))

    # calculate absolute values of speed and acceleration
    data["ego_v"] = np.sqrt(data.ego_vx**2 + data.ego_vy**2)
    data["bot_v"] = np.sqrt(data.bot_vx**2 + data.bot_vy**2)
    data["ego_a"] = np.sqrt(data.ego_ax**2 + data.ego_ay**2)
    data["bot_a"] = np.sqrt(data.bot_ax**2 + data.bot_ay**2)

    # calculate actual distance between the ego vehicle and the bot, and current tta for each t
    data["d_ego_bot"] = np.sqrt((data.ego_x - data.bot_x)**2 + (data.ego_y - data.bot_y)**2)
    data["tta"] = data.d_ego_bot/data.bot_v

    data["truck_v"] = np.sqrt(data.truck_vx**2 + data.truck_vy**2)
    data["angle_diff"] = data.truck_angle - data.bot_angle

    # get the DVs and helper variables
    measures = data.groupby(data.index.names).apply(get_measures)
    print("Number of trials before exclusions: %i" % len(measures))

    # exclude four participants who have very high (>50%) rate of premature responses in "go" trials
    # and one participant who has very high (>50%) proportion of "stay" trials without pressing the yield button
    measures = measures.iloc[~measures.index.get_level_values("subj_id").isin([542, 543, 746, 774])]
    data = data.iloc[~data.index.get_level_values("subj_id").isin([542, 543, 746, 774])]

    # data = data.join(measures)
    measures["is_go_decision"] = measures.min_distance > 5

    # RT_gas is -1 if the bot did not move for some reason or throttle wasn't pressed
    print("Number of go trials without a throttle press or bot not moving: %i" % (len(measures[measures.is_go_decision & (measures.RT_gas==-1)])))

    # RT_yield is -1 if the yield button wasn't pressed in a stay decision
    # Two possible reasons
    # 1) the participant just forgot to press the yield button;
    # 2) this is in fact a go decision just with a very small distance between the cars;
    print("Number of stay trials without a yield button press: %i" % (len(measures[~measures.is_go_decision & (measures.RT_yield==-1)])))
    print(measures[~measures.is_go_decision & (measures.RT_yield==-1)].groupby(["subj_id"]).size())

    # add the condition information to the measures dataframe for further analysis
    conditions = data.loc[:,["tta_condition", "d_condition", "a_condition", "nudge_condition"]].groupby(data.index.names).first()
    measures = measures.join(conditions)

    # add column "decision" for nicer visualization
    measures["decision"] = "Stay"
    measures.loc[measures.is_go_decision, ["decision"]] = "Go"
    measures["p_go_decision"] = measures.decision == "Go"

    measures["RT"] = measures["RT_gas"]
    measures.loc[~measures.is_go_decision, ["RT"]] = measures.loc[~measures.is_go_decision, ["RT_yield"]].values

    print("Number of trials after exclusions: %i" % len(measures))

    return data, measures

data_path = "../data/experiment1-3d"

# merge_csv_files(data_path=data_path)
raw_data = pd.read_csv(os.path.join(data_path, "raw_data_merged.csv"), sep="\t",
                       index_col=["subj_id", "session", "route", "intersection_no"])
processed_data, measures = process_data(raw_data)

measures.to_csv(os.path.join(data_path, "measures.csv"), index=True)
processed_data.to_csv(os.path.join(data_path, "processed_data.csv"), index=True)