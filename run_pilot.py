'''
Our world: an intersection with left-turn scenario
'''

import csv
import os
import random
import time
from datetime import datetime

import scenarios
from intersection_world import IntersectionWorld
from simulator import Simulator


def get_conditions(n_repetitions, fraction_random_trials):
    # create experiment conditions
    d_conditions = [30., 45.]  # distance (m)
    tau_conditions = [4.5]  # TTA (s)
    s_conditions = [0., 0.5, 1.0]  # decision point / states [in seconds from start]
    # a_conditions = [-3.0, 0., 3.0]  # acceleration (m/s2)
    a_combinations = [[0., 0., 0.],
                      [0., 3., 0.],
                      [0., 3., 3.],
                      [0., 3., -3.],
                      [0., -3., 0.],
                      [0., -3., 3.],
                      [0., -3., -3.]]

    conditions = [(d, tau, a, s_conditions, True) # is_test_trial: True
                  for d in d_conditions for tau in tau_conditions for a in a_combinations]
    # all test trials, each condition has n_repetitions
    test_trials = conditions * n_repetitions

    # add extra random trials and randomize
    random_trials = []
    for ii in range(round(fraction_random_trials * len(test_trials))):
        d = random.uniform(d_conditions[0], d_conditions[1])
        tau = random.uniform(2.5, 5.0)
        a = a_combinations[random.randint(0, len(a_combinations)-1)]
        a = [element * random.uniform(0., 2.) for element in a]  # bit of a workaround, if a were a np.array, would've been easier :-)
        random_trials.append((d, tau, a, s_conditions, False)) # is_test_trial: False

    test_trials += random_trials

    random.shuffle(test_trials)

    # add training trials
    # add 2 trials with a car that is (almost) standing still for getting used to the egocar's left-turn movement
    training_trials = [(60, 100., [0., 0., 0.], s_conditions, False)] * 2 + \
                      [(d, tau, a, s_conditions, False) # is_test_trial: False
                       for d in d_conditions for tau in tau_conditions for a in a_combinations]

    # combine
    all_trials = training_trials + test_trials

    return all_trials, training_trials, test_trials


def initialize_log(participant_id):
    log_directory = "data/pilot1"
    log_file_path = os.path.join(log_directory, "participant_" + str(participant_id) + "_"
                                 + datetime.strftime(datetime.now(), "%Y%m%d_%H%M") + ".csv")
    with open(log_file_path, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["participant_id", "d_condition", "tau_condition", "a_condition", "is_test_trial", "decision", "RT", "collision"])
    return log_file_path


def write_log(log_file_path, trial_log):
    with open(log_file_path, "a", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(trial_log)

if __name__ == "__main__":
    # Run an example experiment
    dt = 1. / 50.  # 20 ms time step
    t_end = 5.  # simulation time
    n_rep = 30  # number of repetitions per condition
    fraction_random_trials = 0.2  # fraction of random trials added

    participant_id = input("Enter participant ID: ")
    log_file_path = initialize_log(participant_id=participant_id)

    # create our world
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
    world = IntersectionWorld(dt=dt, width=60., height=110., show_state_text=False)
    all_trials, training_trials, test_trials = get_conditions(n_repetitions=n_rep, fraction_random_trials=fraction_random_trials)

    # specify after which trials to have an automatic break; note: trials start at 0!
    break_after_trial = [11, 190]

    for i, (d_condition, tau_condition, a_condition, s_condition, is_test_trial) in enumerate(all_trials):
        if i < len(training_trials):
            print(f"TRAINING: Trial {i + 1} of {len(training_trials)}")
        else:
            print(f"TEST: Trial {i - len(training_trials) + 1} of {len(test_trials)}")

        # print(f"Distance {d_condition:.0f} m", f"Time gap {tau_condition:.1f} s",
        #       f"Speed {3.6 * d_condition / tau_condition:.2f} km/h", "Acceleration", str(a_condition), "m/s^2",
        #       "Acceleration changes at ", str(s_condition), "s")  # {a_condition[0]:.2f} m/s^2

        # run a scenario in this world
        scenarios.scenario_pilot1(world=world, d0_av=d_condition, v0_av=d_condition / tau_condition, a_av=a_condition, s_av=s_condition)
        sim = Simulator(world, end_time=t_end, ppm=10)

        # run stuff
        kill_switch = sim.run()

        if kill_switch:
            print("Experiment killed")
            break

        # and save stuff (just a proposal for filename coding)
        write_log(log_file_path, [participant_id, int(d_condition), f"{tau_condition:.1f}", str(a_condition), str(is_test_trial),
                                  str(sim.world.agents["human"].decision), f"{sim.world.agents['human'].response_time:.3f}",
                                  str(sim.collision_detected)])  # f"{a_condition:.2f}"

        # clean up
        sim.quit()
        del sim  # not sure if necessary

        time.sleep(0.5)

        # small break
        if i in break_after_trial:
            input("BREAK: Press Enter to continue")

    print("EXPERIMENT DONE (FREEDOM!)")
