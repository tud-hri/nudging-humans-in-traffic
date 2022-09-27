import pyddm
import numpy as np
import models
import loss_functions
import pandas as pd
import os
import utils
from scipy import interpolate
import ast

T_dur = 5.0
breakpoints = np.array([0., 0.25, 1.25, 2.25] + [T_dur])

def get_state_interpolators(tta_0, d_0, a_values):
    v_0 = d_0 / tta_0
    # a_values = np.concatenate([[0], [a_1, a_2], [0., 0.]])
    a_values = np.concatenate([a_values, [0.]])
    v_values = np.concatenate([[v_0], v_0 + np.cumsum(np.diff(breakpoints) * a_values[:-1])])
    d_values = np.concatenate([[d_0], d_0 - np.cumsum(np.diff(breakpoints) * (v_values[1:] + v_values[:-1]) / 2)])
    tta_values = d_values / v_values

    # acceleration is piecewise-constant
    f_a = interpolate.interp1d(breakpoints, a_values, kind=0)
    # under piecewise-constant acceleration, tta is piecewise-linear
    # f_v = interpolate.interp1d(acceleration_timings, v_condition, kind=1, fill_value=(v_0, v_0), bounds_error=False)
    f_tta = interpolate.interp1d(breakpoints, tta_values, kind=1)
    # under piecewise-linear v, d is piecewise-quadratic, but piecewise-linear approximation is very close
    f_d = interpolate.interp1d(breakpoints, d_values, kind=1)

    return f_tta, f_d, f_a


def fit_model(model, training_data, loss_function):
    training_sample = pyddm.Sample.from_pandas_dataframe(df=training_data, rt_column_name="RT",
                                                         correct_column_name="is_go_decision")
    fitted_model = pyddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function,
                                          verbose=False)
    # pyddm.plot.plot_fit_diagnostics(model=fitted_model, sample=training_sample)

    return fitted_model


def fit_model_by_condition(subj_idx=0, loss="vincent"):
    simulation_params = {"dt": 0.1, "n": 50, "duration": T_dur}

    # model = models.ModelTtaDistance()
    # model = models.ModelFixedAcceleration()
    # model = models.ModelAccelerationDependent(
    model = models.ModelAccelerationDependent_v2()

    exp_data = pd.read_csv("data/measures.csv")
    exp_data = exp_data[exp_data.RT < simulation_params["duration"]]
    exp_data.a_condition = exp_data.a_condition.apply(ast.literal_eval).apply(tuple)
    exp_data[["a_0", "a_1", "a_2", "a_3"]] = pd.DataFrame(exp_data["a_condition"].tolist(), index=exp_data.index)

    exp_data = exp_data.rename(columns={"tta_condition": "tta_0",
                                        "d_condition": "d_0"})

    subjects = exp_data.subj_id.unique()

    if subj_idx == "all":
        subj_id = "all"
        subj_data = exp_data
        loss = loss_functions.LossWLSVincent if loss == "vincent" else pyddm.LossRobustBIC
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]
        loss = loss_functions.LossWLS

    output_directory = "fit_results/model_acceleration_dependent_v2"

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
