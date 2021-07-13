'''
Our world: an intersection with left-turn scenario
'''

import csv
import os
import random
import time
from datetime import datetime
from enum import Enum

import numpy as np

from scenarios import ScenarioOpenLoopInteraction
from intersection_world import IntersectionWorld
from simulator import Simulator


class TrialType(Enum):
    TEST = 0
    TRAINING = 1
    RANDOM = 2


def get_conditions(n_repetitions, fraction_random_trials, include_training_trials=True):
    # create experiment conditions
    d_conditions = [30., 45.]  # distance (m)
    tau_conditions = [4.5]  # TTA (s)
    s_conditions = [0., 0.5, 1.0, 1.5]  # decision point / states [in seconds from start]
    a_combinations = [[0., 0., 0., 0.],
                      [0., 3., 0., 0.],
                      [0., 3., 3., 0.],
                      [0., 3., -3., 0.],
                      [0., -3., 0., 0.],
                      [0., -3., 3., 0.],
                      [0., -3., -3., 0.]]

    conditions = [(d, tau, a, s_conditions, TrialType.TEST) for d in d_conditions for tau in tau_conditions for a in a_combinations]

    # all test trials, each condition has n_repetitions
    test_trials = conditions * n_repetitions

    # add extra random trials and randomize
    random_trials = []
    for ii in range(round(fraction_random_trials * len(test_trials))):
        d = round(random.uniform(d_conditions[0], d_conditions[1]), 2)
        tau = round(random.uniform(2.5, 5.0), 2)
        a = a_combinations[random.randint(0, len(a_combinations) - 1)]
        a = [round(element * random.uniform(0., 2.), 2) for element in a]  # bit of a workaround, if a were a np.array, would've been easier :-)
        random_trials.append((d, tau, a, s_conditions, TrialType.RANDOM))

    test_trials += random_trials

    random.shuffle(test_trials)

    # add training trials
    # add 2 trials with a car that is (almost) standing still for getting used to the egocar's left-turn movement
    training_trials = [(60, 100., [0., 0., 0., 0.], s_conditions, TrialType.TRAINING)] * 2 + \
                      [(d, tau, a, s_conditions, TrialType.TRAINING) for d in d_conditions for tau in tau_conditions for a in a_combinations]

    # combine
    if include_training_trials:
        all_trials = training_trials + test_trials
    else:
        all_trials = test_trials

    return all_trials, training_trials, test_trials


def initialize_log(participant_id):
    log_directory = "data/pilot1"
    log_file_path = os.path.join(log_directory, "participant_" + str(participant_id) + "_"
                                 + datetime.strftime(datetime.now(), "%Y%m%d_%H%M") + ".csv")
    with open(log_file_path, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["participant_id", "d_condition", "tau_condition", "a_condition", "is_test_trial", "decision", "RT", "collision", "pet"])
    return log_file_path


def write_log(log_file_path, trial_log):
    with open(log_file_path, "a", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(trial_log)


if __name__ == "__main__":
    # Run an example experiment
    dt = 1. / 60.  # 20 ms time step
    t_end = 5.  # simulation time
    n_rep = 30  # number of repetitions per condition
    fraction_random_trials = 0.05  # fraction of random trials added

    participant_id = input("Enter participant ID: ")
    log_file_path = initialize_log(participant_id=participant_id)

    all_trials, training_trials, test_trials = get_conditions(n_repetitions=n_rep, include_training_trials=True,
                                                              fraction_random_trials=fraction_random_trials)

    # specify after which trials to have an automatic break; note: trials start at 0!
    # three breaks
    break_after_trial = [len(training_trials) - 1] + [int(ii) for ii in np.round(
        len(training_trials) + np.linspace(len(test_trials) / 4, len(test_trials), 3)).tolist()]  # hack hack hack

    # create our world
    # coordinate system: x (right, meters), y (up, meters), psi (CCW, east = 0., rad)
    world = IntersectionWorld(dt=dt, width=60., height=110., show_state_text=False)
    scenario = ScenarioOpenLoopInteraction()
    simulator = Simulator(world=world, end_time=t_end, ppm=10)

    for i, (d_condition, tau_condition, a_condition, s_condition, trial_type) in enumerate(all_trials):
        # scenario creates the agents in the world
        world.reset()  # necessary to reset the post-encroachment time parameters
        scenario.setup_world(world=world, d0_av=d_condition, v0_av=d_condition / tau_condition, a_av=a_condition,
                             s_av=s_condition)

        if trial_type is TrialType.TRAINING:
            print(f"TRAINING: Trial {i + 1} of {len(training_trials)}")
            simulator.user_text = "Training"
        else:
            simulator.user_text = ""
            print(f"TEST: Trial {i - len(training_trials) + 1} of {len(test_trials)}")

        # run stuff
        kill_switch = simulator.run_simulation()

        if kill_switch:
            print("Experiment killed")
            break

        # calculate post encroachment time
        pet = np.round(simulator.world.t_pet_out_av - simulator.world.t_pet_out_human, 4)

        # and save stuff (just a proposal for filename coding)
        write_log(log_file_path, [participant_id, int(d_condition), f"{tau_condition:.1f}", str(a_condition), str(trial_type is TrialType.TEST),
                                  str(simulator.world.agents["human"].decision), f"{simulator.world.agents['human'].response_time:.3f}",
                                  str(simulator.collision_detected), str(pet)])

        # clean up
        # world.clean()
        # scenario.clean_world(world)
        # simulator.quit()
        # del simulator  # not sure if necessary

        time.sleep(0.5)

        # small break
        if i in break_after_trial:
            input("BREAK: Press Enter to continue")

    print("EXPERIMENT DONE (FREEDOM!)")
