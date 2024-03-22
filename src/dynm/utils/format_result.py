"""Auxiliary functions for results formatting."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List


def tidy_parameters(dict_parameters: dict, entry_m: str, entry_v: str,
                    names_parameters: List,
                    index_seas_parameters: List = None,
                    F: np.ndarray = None):
    """Transform the state space moments from dictionary of to a \
    pd.DataFrame in long format.

    Parameters
    ----------
    dict_parameters : dict
        A dictionary with the posterior (m and C) and prior (a and R) moments
        for the state space parameters along time.
    entry_m : str
        An entry name in the `dict_parameters` of mean vector of
        state space parameters.
    entry_v : str
        An entry name in the `dict_parameters` of covariance matrix of
        state space parameters.
    names_parameters : List
        A List with the names of each state parameter components.
    index_seas_parameters : List
        A List indicating the corresponding index for the seasonalities model
        components.
    F : np.ndarray
        An array with the of known constants representing the model components.

    Returns
    -------
    pd.DataFrame
        A DataFrame in tidy format with columns parameter, mean, and
        variance for each time.
    """

    def _get_mean(x: np.ndarray):
        """Get the mean vector from np.nadarray and transform in pd.DataFrame.

        Parameters
        ----------
        x : np.ndarray
            The mean vector of state space parameters. It could be the prior or
            posterior moments.

        Returns
        -------
        pd.DataFrame
            The extracted mean values.
        """
        df_out = pd.DataFrame(
            data={"mean": x[:, 0]}, index=names_parameters)
        if index_seas_parameters:
            j = 1
            lt = []
            for iseas in index_seas_parameters:
                m_seas = x[iseas][:, 0]
                F_seas = F[iseas]
                sum_seas = F_seas.T @ m_seas
                lt.append(pd.DataFrame(
                    data={"mean": sum_seas}, index=["Sum Seas " + str(j)]))
                j = j + 1
            df_out = pd.concat([df_out, pd.concat(lt)])
        return df_out

    def _get_var(x: np.ndarray):
        """Get the diagonal elements of covariance matrix from np.nadarray\
        and transform in pd.DataFrame.

        Parameters
        ----------
        x : np.ndarray
            The covariance matrix of state space parameters.
            It could be the prior or posterior moments.

        Returns
        -------
        pd.DataFrame
            The extracted variance values.
        """
        df_out = pd.DataFrame(
            data={"variance": np.diag(x)}, index=names_parameters)
        if index_seas_parameters:
            j = 1
            lt = []
            for iseas in index_seas_parameters:
                cov_seas = x[np.ix_(iseas, iseas)]
                F_seas = F[iseas]
                sum_seas = F_seas.T @ cov_seas @ F_seas
                lt.append(pd.DataFrame(
                    data={"variance": sum_seas[:, 0]},
                    index=["Sum Seas " + str(j)]))
                j = j + 1
            df_out = pd.concat([df_out, pd.concat(lt)])

        return df_out

    df_mean_parms = pd.concat(
        list(map(_get_mean, dict_parameters[entry_m])))
    df_var_parms = pd.concat(
        list(map(_get_var, dict_parameters[entry_v])))
    df_state_parameters = pd.concat(
        [df_mean_parms.reset_index(),
         df_var_parms.reset_index(drop=True)], axis=1)
    df_state_parameters.rename(columns={"index": "parameter"}, inplace=True)

    return df_state_parameters[["parameter", "mean", "variance"]]


def add_credible_interval_studentt(
        pd_df: pd.DataFrame,
        entry_m: str,
        entry_v: str,
        level=float):
    df = pd_df["t"].values + 1
    mu = pd_df[entry_m].values
    sigma = np.sqrt(pd_df[entry_v].values + 10e-300)

    # Calculate intervals
    pd_df["ci_lower"] = stats.t.ppf(q=level/2, df=df, loc=mu, scale=sigma)
    pd_df["ci_upper"] = stats.t.ppf(q=1-level/2, df=df, loc=mu, scale=sigma)

    return pd_df


def add_credible_interval_gamma(
        pd_df: pd.DataFrame,
        entry_a: str,
        entry_b: str,
        level=float):
    a = 1 / pd_df[entry_a].values
    b = pd_df[entry_b].values + 10e-300

    # Calculate intervals
    pd_df["ci_lower"] = stats.gamma.ppf(q=level/2, a=a, scale=b)
    pd_df["ci_upper"] = stats.gamma.ppf(q=1-level/2, a=a, scale=b)

    return pd_df


def create_mod_label_column(mod, t: int):
    dlm_model_idx = mod.dlm.model_index_dict
    dnm_model_idx = mod.dnm.model_index_dict

    dlm_lb = np.concatenate([np.repeat(k, len(dlm_model_idx.get(k)))
                             for k in dlm_model_idx.keys()])
    dnm_lb = np.concatenate([np.repeat(k, len(dnm_model_idx.get(k)))
                             for k in dnm_model_idx.keys()])

    mod_lb = t * list(np.concatenate([dlm_lb, dnm_lb]))

    return mod_lb


def _build_predictive_df(mod, dict_predict: dict, level: float):
    df_predictive = pd.DataFrame(dict_predict)

    # Compute credible intervals
    df_predictive = add_credible_interval_studentt(
        pd_df=df_predictive, entry_m="f",
        entry_v="q", level=.05)

    return df_predictive


def _build_posterior_df(
        mod,
        dict_posterior: dict,
        entry_m: str,
        entry_v: str,
        t: int,
        level: float,
        smooth: bool = False):
    # Organize the posterior parameters
    df_posterior = tidy_parameters(
        dict_parameters=dict_posterior,
        entry_m=entry_m, entry_v=entry_v,
        names_parameters=mod.names_parameters)

    # Create model labels
    df_posterior["mod"] = create_mod_label_column(mod=mod, t=t)

    # Add time column on posterior_df
    if smooth:
        t_index = mod.t - np.arange(0, mod.t)
    else:
        t_index = np.arange(1, t+1)

    df_posterior["t"] = np.repeat(t_index, mod.m.shape[0])
    df_posterior["t"] = df_posterior["t"].astype(int)

    # Round variance
    df_posterior["variance"] = df_posterior["variance"].round(10)

    # Compute credible intervals
    df_posterior = add_credible_interval_studentt(
        pd_df=df_posterior, entry_m="mean",
        entry_v="variance", level=.05)

    return df_posterior


def _build_variance_df(
        mod,
        dict_observation_var: dict,
        level: float):

    # Organize observational variance
    df_var = pd.DataFrame(dict_observation_var)\
        .assign(
            variance=lambda x: x.d / (x.n ** 2),
            parameter="V",
            mod="observational_variance"
    )

    # Organize observational variance
    df_var = add_credible_interval_gamma(
        pd_df=df_var, entry_a="n", entry_b="d", level=level)
    df_var.drop(['d', 'n'], axis=1, inplace=True)

    return df_var
