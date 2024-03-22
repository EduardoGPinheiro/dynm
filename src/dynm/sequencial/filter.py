"""Filtering for Dynamic Linear Models."""
import numpy as np
import pandas as pd
from dynm.utils.format_result import _build_predictive_df, _build_posterior_df
from dynm.utils.format_result import _build_variance_df
from dynm.utils.format_input import set_X_dict


def _foward_filter(mod,
                   y: np.ndarray,
                   X: dict = {},
                   level: float = 0.05):
    """Short summary.

    Parameters
    ----------
    y : np.ndarray
        Description of parameter `y`.
    x : np.ndarray
        Description of parameter `x`.

    Returns
    -------
    type
        Description of returned object.

    """
    nobs = len(y)

    dict_1step_forecast = {'t': [], 'y': [], 'f': [], 'q': []}
    dict_observation_var = {'t': [], 'd': [], 'n': [], 'mean': []}
    dict_state_params = {'m': [], 'C': [], 'a': [], 'R': []}
    dict_state_evolution = {'G': []}

    Xt = {'regression': [], 'transfer_function': []}
    copy_X = set_X_dict(mod=mod, nobs=nobs, X=X)

    for t in range(nobs):
        # Predictive distribution moments
        Xt['regression'] = copy_X['regression'][t, :]
        Xt['transfer_function'] = copy_X['transfer_function'][t, :, :]

        # Update model
        mod._update(y=y[t], X=Xt)

        # Dict 1steap forecast
        dict_1step_forecast['t'].append(t+1)
        dict_1step_forecast['y'].append(y[t])
        dict_1step_forecast['f'].append(mod.f)
        dict_1step_forecast['q'].append(mod.q)

        # Dict state params
        dict_state_params["a"].append(mod.a)
        dict_state_params["R"].append(mod.R)
        dict_state_params["m"].append(mod.m)
        dict_state_params["C"].append(mod.C)

        # State evolution matrix
        dict_state_evolution['G'].append(mod.G)

        # Observational variance
        dict_observation_var['t'].append(t+1)
        dict_observation_var['d'].append(np.ravel(mod.d)[0])
        dict_observation_var['n'].append(np.ravel(mod.n)[0])
        dict_observation_var['mean'].append(np.ravel(mod.s)[0])

    # Get posterior and predictive dataframes
    df_predictive = _build_predictive_df(
        mod=mod, dict_predict=dict_1step_forecast, level=level)

    df_posterior = _build_posterior_df(
        mod=mod,
        dict_posterior=dict_state_params,
        entry_m="m",
        entry_v="C",
        t=nobs,
        level=level)

    df_var = _build_variance_df(
        mod=mod,
        dict_observation_var=dict_observation_var,
        level=level)

    # Concatenate results
    df_posterior = pd.concat([df_posterior, df_var])\
        .sort_values(['t', 'parameter'])

    filter_dict = {
        "predictive": df_predictive,
        "posterior": df_posterior
    }

    # Creat dict of results
    return_dict = {
        "filter": filter_dict,
        "state_params": dict_state_params,
        "state_evolution": dict_state_evolution,
        "fitted_mod": mod
    }

    return return_dict
