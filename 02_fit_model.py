import pyddm
import numpy as np
import models
import loss_functions
import pandas as pd
import os
import utils
from scipy import interpolate
import ast

def fit_model(model, training_data, loss_function):
    training_sample = pyddm.Sample.from_pandas_dataframe(df=training_data, rt_column_name="RT",
                                                         correct_column_name="is_go_decision")
    fitted_model = pyddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function,
                                          verbose=True)

    return fitted_model


def fit_model_by_condition(subj_idx=0, loss="vincent", T_dur=6):
    model = models.ModelAccelerationDependent()

    exp_data = pd.read_csv("data/measures.csv")

    # This excludes very few trials with long RTs, but also excludes about 800 (21%) trials with missing RTs (unless they are replaced by 0 already)
    exp_data = exp_data[(exp_data.RT < T_dur)]

    exp_data.a_values = exp_data.a_values.apply(ast.literal_eval).apply(tuple)

    # training on a subset of data
    exp_data = exp_data[(exp_data.a_values == (0.0, 4, 4, 0.0))
                        | (exp_data.a_values == (0.0, 0.0, 0.0, 0.0))
                        | (exp_data.a_values == (0.0, -4, -4, 0.0))]

    subjects = exp_data.subj_id.unique()

    if subj_idx == "all":
        subj_id = "all"
        subj_data = exp_data
        loss = loss_functions.LossWLSVincent if loss == "vincent" else pyddm.LossRobustBIC
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]
        loss = loss_functions.LossWLS

    training_data = subj_data
    print("len(training_data): " + str(len(training_data)))

    output_directory = "modeling/fit_results_excluded_nan_rt/model_acceleration_dependent_cross_validation"

    file_name = "subj_%s_parameters_fitted.csv" % (str(subj_id))
    if not os.path.isfile(os.path.join(output_directory, file_name)):
        utils.write_to_csv(output_directory, file_name, ["subj_id", "loss"] + model.param_names, write_mode="w")

    print(subj_id)

    fitted_model = fit_model(model.model, training_data, loss)
    utils.write_to_csv(output_directory, file_name,
                       [subj_id, fitted_model.get_fit_result().value()]
                       + [float(param) for param in fitted_model.get_model_parameters()])

    return fitted_model

fitted_model = fit_model_by_condition(subj_idx="all", loss="robustBIC", T_dur=6)
