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
    # model = models.ModelTtaDistance()
    # model = models.ModelFixedAcceleration()
    model = models.ModelAccelerationDependent()
    # model = models.ModelAccelerationDependent_v2()

    exp_data = pd.read_csv("data/measures.csv")

    # exclude four participants who have very high (>50%) rate of premature responses in "go" trials
    # and one participant who has very high (>50%) proportion of "stay" trials without pressing the yield button
    # and then replaces all remaining missing RTs (12% go, 4% stay) with 0, so skews the RT estimates towards 0, but retains p(go)
    exp_data = exp_data[~exp_data.subj_id.isin([542, 543, 746, 774])]
    exp_data["RT"] = exp_data["RT"].fillna(0)

    # This excludes very few trials with long RTs, but also excludes about 800 (21%) trials with missing RTs (unless they are replaced by 0 already)
    exp_data = exp_data[(exp_data.RT < T_dur)]

    exp_data.a_values = exp_data.a_values.apply(ast.literal_eval).apply(tuple)
    # exp_data[["a_0", "a_1", "a_2", "a_3"]] = pd.DataFrame(exp_data["a_condition"].tolist(), index=exp_data.index)

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

    output_directory = "modeling/fit_results_replaced_nan_rt/model_acceleration_dependent_cross_validation"

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

# print(pd.read_csv("../data/measures.csv"))
fitted_model = fit_model_by_condition(subj_idx="all", loss="robustBIC", T_dur=6)
