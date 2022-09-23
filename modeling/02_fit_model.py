import pyddm
import numpy as np
import models
import loss_functions
import pandas as pd
import os
import utils

def fit_model(model, training_data, loss_function):
    training_sample = pyddm.Sample.from_pandas_dataframe(df=training_data, rt_column_name="RT",
                                                         correct_column_name="is_go_decision")
    fitted_model = pyddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function, verbose=False)
    # pyddm.plot.plot_fit_diagnostics(model=fitted_model, sample=training_sample)

    return fitted_model

def fit_model_by_condition(subj_idx=0, loss="vincent"):
    simulation_params = {"dt": 0.1, "n": 40, "duration": 5.0}

    # Problem 1: get tta, d, a interpolators for a single condition

    # Problem 2: loop over all condition combinations and pack all interpolators in a dictionary or smth

    # interpolators = {condition: value for (condition, value) in data}
     # = [[f_tta, f_d, f_a] for condition ]
    # then this function will just look up the
    def f_get_env_state(t, conditions):
        f_tta, f_d, f_a = interpolators[conditions]
        tta = f_tta(t)
        d = f_d(t)
        a = f_a(t)
        return tta, d, a

    # model = models.ModelTtaDistance()
    model = models.ModelFixedAcceleration(f_get_env_state=f_get_env_state)
    # model = models.ModelAccelerationDependent(simulation_params["n"])

    exp_data = pd.read_csv("../data/experiment1-3d/measures.csv")
    exp_data = exp_data[exp_data.RT < simulation_params["duration"]]
    subjects = exp_data.subj_id.unique()

    if subj_idx == "all":
        subj_id = "all"
        subj_data = exp_data
        loss = loss_functions.LossWLSVincent if loss == "vincent" else pyddm.LossRobustBIC
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]
        loss = loss_functions.LossWLS

    output_directory = "fit_results/drift_tta_distance"

    file_name = "subj_%s_parameters_fitted.csv" % (str(subj_id))
    if not os.path.isfile(os.path.join(output_directory, file_name)):
        utils.write_to_csv(output_directory, file_name, ["subj_id", "loss"] + model.param_names, write_mode="w")

    print(subj_id)

    training_data = subj_data
    print("len(training_data): " + str(len(training_data)))

    fitted_model = fit_model(model.model, training_data, loss)
    utils.write_to_csv(output_directory, file_name,
                        [subj_id, fitted_model.get_fit_result().value()]
                        + [float(param) for param in fitted_model.get_model_parameters()])

    return fitted_model


fitted_model = fit_model_by_condition(subj_idx="all", loss="robustBIC")


