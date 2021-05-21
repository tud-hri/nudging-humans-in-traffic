'''
Our world: an intersection with left-turn scenario
'''

from datetime import datetime
import os
import scenarios
import random
import csv
from intersection_world import IntersectionWorld
from simulator import Simulator


def get_conditions(n_repetitions):
    # create experiment conditions (just some numbers)
    d_conditions = [60., 90.]
    v_conditions = [50. / 3.6, 75. / 3.6]
    a_conditions = [-0.3, 0., 0.3]

    conditions = ([(d, v, a) for d in d_conditions for v in v_conditions for a in a_conditions] * n_rep)

    random.shuffle(conditions)
    return conditions


def initialize_log(participant_id):
    log_directory = "data"
    log_file_path = os.path.join(log_directory, "participant_" + str(participant_id) + "_"
                                 + datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M") + ".csv")
    with open(log_file_path, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["participant_id", "d_condition", "v_condition", "a_condition", "decision", "RT"])
    return log_file_path


def write_log(log_file_path, trial_log):
    with open(log_file_path, "a", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(trial_log)


if __name__ == "__main__":
    # Run an example experiment
    dt = 1. / 60.  # 16.67 ms time step
    n_rep = 10  # number of repetitions per condition

    participant_id = input("Enter participant ID: ")
    log_file_path = initialize_log(participant_id=participant_id)

    # create our world
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
    world = IntersectionWorld(dt=dt, width=60., height=120.)
    conditions = get_conditions(n_repetitions=n_rep)

    for i, (d_condition, v_condition, a_condition) in enumerate(conditions):
        print(i)
        print(d_condition, v_condition, a_condition)

        # run a scenario in this world
        scenarios.scenario_pilot1(world=world, d0_av=d_condition, v0_av=v_condition, a_av=a_condition)
        sim = Simulator(world, end_time=5., ppm=8)

        # run stuff
        sim.run()
        # and save stuff (just a proposal for filename coding)
        # sim.save_stuff(filename="d{0:d}_v{1:d}_a{2:d}".format(int(d_condition), int(v_condition), int(a_condition)))
        write_log(log_file_path, [participant_id, d_condition, v_condition, a_condition,
                                  sim.world.agents["human"].decision, sim.world.agents["human"].response_time])
