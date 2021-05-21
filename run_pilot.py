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
    d_conditions = [30., 60.]
    tau_conditions = [3.5, 5.]
    a_conditions = [-2.0, 0., 4.0]

    conditions = ([(d, tau, a) for d in d_conditions for tau in tau_conditions for a in a_conditions] * n_repetitions)

    random.shuffle(conditions)
    return conditions


def initialize_log(participant_id):
    log_directory = "data"
    log_file_path = os.path.join(log_directory, "participant_" + str(participant_id) + "_"
                                 + datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M") + ".csv")
    with open(log_file_path, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["participant_id", "d_condition", "tau_condition", "a_condition", "decision", "RT"])
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
    world = IntersectionWorld(dt=dt, width=60., height=110.)
    conditions = get_conditions(n_repetitions=n_rep)

    for i, (d_condition, tau_condition, a_condition) in enumerate(conditions):
        print(f"Trial {i}")
        print(f"Distance {d_condition:.0f} m", f"Time gap {tau_condition:.1f} s",
              f"Speed {3.6*d_condition/tau_condition:.2f} km/h", f"Acceleration {a_condition:.2f} m/s^2")

        # run a scenario in this world
        scenarios.scenario_pilot1(world=world, d0_av=d_condition, v0_av=d_condition/tau_condition, a_av=a_condition)
        sim = Simulator(world, end_time=5., ppm=12)

        # run stuff
        kill_switch = sim.run()

        if kill_switch:
            print("Experiment killed")
            break

        # and save stuff (just a proposal for filename coding)
        write_log(log_file_path, [participant_id, int(d_condition), f"{tau_condition:.1f}", f"{a_condition:.2f}",
                                  str(sim.world.agents["human"].decision), f"{sim.world.agents['human'].response_time:.3f}"])
