import numpy as np
from scipy import interpolate, optimize
import pyddm
import pandas as pd


class LossWLS(pyddm.LossFunction):
    name = "Weighted least squares as described in Ratcliff & Tuerlinckx 2002"
    rt_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    rt_q_weights = [2, 2, 1, 1, 0.5]

    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur

    def get_rt_quantiles(self, cdf, t_domain, exp=False, correct=True):
        # print(cdf)
        cdf_interp = interpolate.interp1d(t_domain, cdf / cdf[-1])

        # If the model produces very fast RTs, interpolated cdf(0) can be >0.1, then we cannot find root like usual
        # In this case, the corresponding rt quantile is half of the time step of cdf
        rt_quantile_values = [optimize.root_scalar(lambda x: cdf_interp(x) - quantile, bracket=(0, t_domain[-1])).root
                              if (cdf_interp(0) < quantile) else self.dt / 2
                              for quantile in self.rt_quantiles]
        return np.array(rt_quantile_values)

    def loss(self, model):
        solutions = self.cache_by_conditions(model)
        WLS = 0
        for condition in self.sample.condition_combinations(required_conditions=self.required_conditions):
            c = frozenset(condition.items())
            #            print(c)
            condition_data = self.sample.subset(**condition)
            WLS += 4 * (solutions[c].prob_correct() - condition_data.prob_correct()) ** 2

            # These are needed for vincentized distributions
            # TODO: fix this so that this logic is a part of the child class
            [self.condition_rts_correct, self.condition_rts_error] = \
                [pd.DataFrame([[item[0], item[1]["subj_id"]] for item in condition_data.items(correct=correct)],
                              columns=["RT", "subj_id"]) for correct in [True, False]]

            # Sometimes model p_correct is very close to 0, then RT distribution is weird, in this case ignore RT mismatch
            if ((solutions[c].prob_correct() > 0.001) & (condition_data.prob_correct() > 0)):
                model_rt_q_corr = self.get_rt_quantiles(solutions[c].cdf_corr(), model.t_domain(), exp=False,
                                                        correct=True)
                # for vincentized loss, these calls to self.get_rt_quantiles for exp=True are redirected to the child class
                exp_rt_q_corr = self.get_rt_quantiles(condition_data.cdf_corr(T_dur=self.T_dur, dt=self.dt),
                                                      model.t_domain(), exp=True, correct=True)

                WLS += np.dot((model_rt_q_corr - exp_rt_q_corr) ** 2, self.rt_q_weights) * condition_data.prob_correct()

            if ((solutions[c].prob_error() > 0.001) & (condition_data.prob_error() > 0)):
                model_rt_q_error = self.get_rt_quantiles(solutions[c].cdf_err(), model.t_domain(), exp=False,
                                                         correct=False)

                exp_rt_q_error = self.get_rt_quantiles(condition_data.cdf_err(T_dur=self.T_dur, dt=self.dt),
                                                       model.t_domain(), exp=True, correct=False)

                WLS += np.dot((model_rt_q_error - exp_rt_q_error) ** 2, self.rt_q_weights) * condition_data.prob_error()
        return WLS


class LossWLSVincent(LossWLS):
    name = """Weighted least squares as described in Ratcliff & Tuerlinckx 2002, 
                fitting to the quantile function vincent-averaged per subject (Ratcliff 1979)"""

    def get_rt_quantiles(self, cdf, t_domain, exp=False, correct=True):
        # hack: for vincentized loss, cdf is not even used, we use self.condition_rts then
        if exp:
            condition_rts = self.condition_rts_correct if correct else self.condition_rts_error

            vincentized_quantiles = (condition_rts.groupby("subj_id")
                                     .apply(lambda group: np.quantile(a=group.RT, q=self.rt_quantiles))).mean()
            return vincentized_quantiles
        else:
            return super().get_rt_quantiles(cdf, t_domain, exp=exp, correct=correct)
