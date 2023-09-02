import pyddm
import models
import loss_functions
import pandas as pd
import os
import utils
from datetime import datetime
import ast

def fit_model(model, training_data, loss_function):
    training_sample = pyddm.Sample.from_pandas_dataframe(df=training_data, rt_column_name="RT",
                                                         choice_column_name="is_go_decision", choice_names=("Go", "Stay"))
    fitted_model = pyddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function, verbose=True)

    return fitted_model


def fit_model_by_condition(model_no=1, subj_idx=0, loss_name="bic", T_dur=4):
    model = models.get_model(model_no=model_no, T_dur=T_dur)
    exp_data = pd.read_csv("data/measures.csv")
    # This excludes a small fraction of trials with outlier RTs, but also excludes about 800 trials with missing RTs (unless they are replaced by 0 already)
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
    else:
        subj_id = subjects[subj_idx]
        subj_data = exp_data[(exp_data.subj_id == subj_id)]

    training_data = subj_data
    print("len(training_data): " + str(len(training_data)))

    if loss_name == "bic":
        loss = pyddm.LossRobustBIC
    elif loss_name=="vincent":
        loss = loss_functions.LossWLSVincent
    elif loss_name == "wls":
        loss = loss_functions.LossWLS
    else:
        raise Exception("Loss name not recognized")

    output_directory = "modeling/fit_results_%s/model_%i" % (loss_name, model_no)

    file_name = "subj_%s_parameters_fitted.csv" % (str(subj_id))
    if not os.path.isfile(os.path.join(output_directory, file_name)):
        utils.write_to_csv(output_directory, file_name, ["subj_id", "loss"] + model.get_model_parameter_names(), write_mode="w")

    print(subj_id)

    fitted_model = fit_model(model, training_data, loss)
    utils.write_to_csv(output_directory, file_name,
                       [subj_id, fitted_model.get_fit_result().value()]
                       + [float(param) for param in fitted_model.get_model_parameters()])

    with open("modeling/logs/%s_model_%i_%s.txt" % (loss_name, model_no, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), "w") as outfile:
        print(fitted_model, file=outfile)

    return fitted_model

fitted_model = fit_model_by_condition(model_no=1, subj_idx="all", loss_name="bic", T_dur=4)
