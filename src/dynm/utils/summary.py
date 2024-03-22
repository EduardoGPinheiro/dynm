"""Summary functions for model results."""
import numpy as np
from scipy import stats


def summary(mod):
    nobs = mod.t

    # Return last time posterior parameters
    print_filter_tab = mod.dict_filter\
        .get('posterior').query("t==@nobs").copy()\
        .reset_index(drop=True)\
        .sort_values(['mod', 'parameter'])

    # Get log-likelihood
    llk = get_predictive_log_likelihood(mod=mod)

    # Print the summary
    summary = "Bayesian Dynamic Linear Model Results\n\n"
    summary += f"Posterior parameters estimate at time {nobs}\n\n"
    summary += str(print_filter_tab) + "\n\n"
    summary += f"Predictive log-likelihood {llk}\n\n"

    # Return both the results and the captured output
    return summary


def get_predictive_log_likelihood(mod):
    predictive_df = mod.dict_filter.get('predictive').dropna().copy()
    y = predictive_df.y.values
    f = predictive_df.f.values
    q = np.sqrt(predictive_df.q.values)
    t = predictive_df.t.values

    llk = np.log(np.sum(stats.t.pdf(x=y, df=t+1, loc=f, scale=q)))
    return llk
