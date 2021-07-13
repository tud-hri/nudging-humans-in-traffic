from modeling import human_models
import ddm
import pandas as pd
import paranoid as pns
pns.settings.Settings.set(enabled=False)

data = pd.read_csv('data/pilot0/participant_1_2021_05_22_20_32.csv', sep='\t',
                   usecols=["participant_id", "d_condition", "tau_condition", "a_condition", "decision", "RT"])

data["is_turn_decision"] = data.decision == "go"
data = data.drop("decision", axis=1)
data = data[(data.RT>0) & (data.RT<3)]

conditions = [{"d": d, "tau": tau, "a": a}
              for d in data.d_condition.unique()
              for tau in data.tau_condition.unique()
              for a in data.a_condition.unique()]

training_sample = ddm.Sample.from_pandas_dataframe(df=data,
                                                   rt_column_name='RT',
                                                   correct_column_name='is_turn_decision')

model = human_models.HumanModelDDMDynamicDriftFittable()

fitted_model = ddm.fit_adjust_model(sample=training_sample, model=model.model, suppress_output=False)
print(fitted_model.get_fit_result().value())
print(fitted_model.get_model_parameters())