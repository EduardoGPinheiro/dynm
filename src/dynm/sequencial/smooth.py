"""Smoothing for Dynamic Linear Models."""
import numpy as np
from dynm.utils.format_result import _build_predictive_df, _build_posterior_df
from dynm.utils.format_input import set_X_dict


def _backward_smoother(mod, X: dict = {}, level: float = 0.05):
    """Perform backward smoother.

    That is, obtain the smoothing moments of the one-step ahead predictive
    distribution and state space posterior distribution.

    Parameters
    ----------
    dict_state_parms : dict
        dictionary with the posterior (m and C) and prior (a and R) moments
        for the state space parameters along time.

    Returns
    -------
    List: It contains the following components:
        - `df_predictive_smooth`: pd.DataFrame with the smoothing moments
        of predictive distribution.

        - `df_posterior_smooth`: pd.DataFrame with the smoothing moments
        of posterior state space distribution.
    """
    nobs = len(mod.dict_state_params.get('a'))
    copy_X = set_X_dict(mod=mod, nobs=nobs, X=X)

    # Initialize the model components and posterior/prior parameters
    a = mod.dict_state_params.get('a')
    R = mod.dict_state_params.get('R')
    m = mod.dict_state_params.get('m')
    C = mod.dict_state_params.get('C')

    # Get state evolution matrix
    G = mod.dict_state_evolution.get('G')

    # Dictionaty to save predictive and posterior parameters
    Xk = {'regression': [], 'transfer_function': []}
    Xk['regression'] = copy_X['regression'][nobs-1, :]
    Xk['transfer_function'] = copy_X['transfer_function'][nobs-1, :]

    FT = mod._build_F(x=Xk['regression'])

    ak = m[nobs-1]
    Rk = C[nobs-1]
    fk = FT.T @ ak
    qk = (FT.T @ Rk @ FT).round(10)
    dict_smooth_params = {
        "t": [nobs],
        "a": [ak],
        "R": [Rk],
        "f": [fk.item()],
        "q": [qk.item()]}

    # Perform smoothing
    for k in range(1, nobs):
        Xk['regression'] = copy_X['regression'][nobs-k-1, :]
        Xk['transfer_function'] = copy_X['transfer_function'][nobs-k-1, :, :]

        Fk = mod._build_F(x=Xk['regression'])
        Gk = G[nobs-k]

        # B_{t-k}
        B_t_k = C[nobs-k-1] @ Gk.T @ np.linalg.pinv(
            R[nobs-k], rcond=1e-10, hermitian=True)

        # a_t(-k) and R_t(-k)
        ak = m[nobs-k-1] + B_t_k @ (ak - a[nobs-k])
        Rk = C[nobs-k-1] + B_t_k @ (Rk - R[nobs-k]) @ B_t_k.T

        # f_t(-k) and q_t(-k)
        fk = Fk.T @ ak
        qk = (Fk.T @ Rk @ Fk).round(10)

        # Saving parameters
        dict_smooth_params["a"].append(ak)
        dict_smooth_params["R"].append(Rk)
        dict_smooth_params["f"].append(fk.item())
        dict_smooth_params["q"].append(qk.item())
        dict_smooth_params["t"].append(nobs-k)

    # Organize the predictive smooth parameters
    dict_filter = {key: dict_smooth_params[key] for key in (
        dict_smooth_params.keys() & {"t", "f", "q", "df"})}

    # Get posterior and predictive dataframes
    df_predictive = _build_predictive_df(
        mod=mod, dict_predict=dict_filter, level=level)

    df_posterior = _build_posterior_df(
        mod=mod,
        dict_posterior=dict_smooth_params,
        entry_m="a",
        entry_v="R",
        t=nobs,
        level=level,
        smooth=True)

    # Creat dict of results
    smooth_dict = {'predictive': df_predictive, 'posterior': df_posterior}

    return_dict = {
        "smooth": smooth_dict,
        "smooth_params": dict_smooth_params
    }

    return return_dict
