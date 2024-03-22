"""Utils functions."""
import numpy as np
import pandas as pd
from copy import deepcopy as copy
from dynm.superposition_block.dlm import DynamicLinearModel
from dynm.superposition_block.dnm import DynamicNonLinearModel
from scipy.linalg import block_diag
from dynm.utils.summary import summary
from dynm.utils.format_result import _build_predictive_df, _build_posterior_df
from dynm.utils.format_input import set_X_dict
from dynm.sequencial.filter import _foward_filter
from dynm.sequencial.smooth import _backward_smoother
from dynm.utils.summary import get_predictive_log_likelihood
from dynm.utils import validation


class BayesianDynamicModel():
    """Class for fitting, forecast and update bayesian dynamic models."""

    def __init__(self, model_dict: dict, V: float = None):
        """Define model.

        Define model with observation/system equations components
        and initial information for prior moments.

        Parameters
        ----------
        model_dict : dict
            Dictionary containing prior moments and other model definition
            parameters for each structural block: polynomial, regression,
            seasonal, transfer function and autoregressive.

            Each structural block should be identified by its label and
            should have the following parameters:

            polynomial:
                Obrigatory keys: {'m0', 'C0', 'order'}.

                Optional keys (choose one): {'W', 'discount'}.

                - 'm0' (np.ndarray): Prior mean vector for the
                   polynomial model components.

                - 'C0' (np.ndarray): Prior covariance matrix for
                   the polynomial model components.

                - 'ntrend' (int): Number of trend components in the
                   polynomial model.

                - 'W' (np.ndarray, optional): Process noise covariance matrix.
                   If 'W' is unkown it will be estimated
                   using discount factor.

                - 'discount' (float, optional): Discount factor
                   betwen 0 and 1. If 'W' is unkown it will be estimated
                   using discount factor.

            regression:
                Obrigatory keys: {'m0', 'C0', 'nregn'}.

                Optional keys (choose one): {'W', 'discount'}.

                - 'm0' (np.ndarray): Prior mean vector for the regression
                   model components.

                - 'C0' (np.ndarray): Prior covariance matrix for the
                   regression model components.

                - 'nregn' (int): Number of regression components in the
                   regression model.

                - 'W' (np.ndarray, optional): Process noise covariance matrix.
                   Choose either 'W' or 'discount'.

                - 'discount' (float, optional): Discount factor.
                   Choose either 'W' or 'discount'.

            seasonal:
                Obligatory keys: {'m0', 'C0',
                                  'seas_period', 'seas_harm_components'}.

                Optional keys (choose one): {'W', 'discount'}.

                - 'm0' (np.ndarray): Prior mean vector for the seasonal
                   model components.

                - 'C0' (np.ndarray): Prior covariance matrix for the
                   seasonal model components.

                - 'seas_period' (float): Period of the seasonal pattern.

                - 'seas_harm_components' (list): List of harmonic components
                   for the seasonal pattern.

                - 'W' (np.ndarray, optional): Process noise covariance matrix.
                   Choose either 'W' or 'discount'.

                - 'discount' (float, optional): Discount factor. Choose
                   either 'W' or 'discount'.

            transfer_function:
                Obligatory keys: {'m0', 'C0', 'ntfm',
                                  'lambda_order', 'gamma_order'}.

                Optional keys (choose one): {'W', 'discount'}.

                - 'm0' (np.ndarray): Prior mean vector for the transfer
                   function model components.

                - 'C0' (np.ndarray): Prior covariance matrix for the
                   transfer function model components.

                - 'ntfm' (int): Number of transfer function blocks.

                - 'lambda_order' (int): Order of the autoregressive component
                   of the transfer function.

                - 'gamma_order' (int): Order of the moving average component
                   of the transfer function.

                - 'W' (np.ndarray, optional): Process noise covariance matrix.
                   Choose either 'W' or 'discount'.

                - 'discount' (float, optional): Discount factor.
                   Choose either 'W' or 'discount'.

            autoregressive:
                Obligatory keys: {'m0', 'C0', 'order'}.

                Optional keys (choose one): {'W', 'discount'}.

                - 'm0' (np.ndarray): Prior mean vector for the
                   autoregressive model components.

                - 'C0' (np.ndarray): Prior covariance matrix for the
                   autoregressive model components.

                - 'order' (int): Order of the autoregressive model.

                - 'W' (np.ndarray, optional): Process noise covariance matrix.
                   Choose either 'W' or 'discount'.

                - 'discount' (float, optional): Discount factor. Choose either
                   'W' or 'discount'.

        """
        self.model_dict = copy(model_dict)
        self.V = V

        self._set_superposition_blocks()

        self._set_gamma_distribution_parameters()

        self._concatenate_regression_vector()

        self._concatenate_evolution_matrix()

        self._concatenate_prior_mean()

        self._concatenate_prior_covariance_matrix()

        self._set_superposition_block_index()

        self._set_parameters_name()

    def _set_superposition_blocks(self):
        """
        Set the superposition blocks.

        Set the superposition blocks for both DynamicLinearModel (DLM) and
        DynamicNonLinearModel (DNM).
        """
        dlm = DynamicLinearModel(model_dict=self.model_dict)
        dnm = DynamicNonLinearModel(model_dict=self.model_dict, V=self.V)

        self.dlm = dlm
        self.dnm = dnm

    def _set_gamma_distribution_parameters(self):
        self.n = 1
        self.t = 0

        if self.V is None:
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.d = 0
            self.s = self.V
            self.estimate_V = False

        if self.dnm.autoregressive_model.order > 0:
            self.v = 0
        else:
            self.v = self.s

    def _concatenate_regression_vector(self):
        """
        Concatenate regression vectors.

        Concatenates the regression vectors from DLM and DNM into a
        single vector 'F'.
        """
        self.F = np.vstack((self.dlm.F, self.dnm.F))

    def _concatenate_evolution_matrix(self):
        """
        Concatenate equation evolution matrices.

        Concatenates the evolution matrices from DLM and DNM into a
        single matrix 'G'.
        """
        self.G = block_diag(self.dlm.G, self.dnm.G)

    def _concatenate_prior_mean(self):
        """
        Concatenate prior mean vectors.

        Concatenates the prior mean vectors from DLM and DNM into a
        single vector 'a'.
        """
        self.a = np.vstack((self.dlm.m, self.dnm.m))
        self.m = np.vstack((self.dlm.m, self.dnm.m))

    def _concatenate_prior_covariance_matrix(self):
        """
        Concatenate prior covariance matrices.

        Concatenates the prior covariance matrices from DLM and DNM into a
        single matrix 'R'.
        """
        self.R = block_diag(self.dlm.C, self.dnm.C)
        self.C = block_diag(self.dlm.C, self.dnm.C)

    def _set_superposition_block_index(self):
        """
        Set superposition block indices.

        Sets the indices for the superposition blocks in the concatenated
        vectors/matrices.
        """
        nparams_dlm = len(self.dlm.m)
        nparams_dnm = len(self.dnm.m)

        block_idx = np.cumsum([nparams_dlm, nparams_dnm])

        idx_dlm = np.arange(0, block_idx[0])
        idx_dnm = np.arange(block_idx[0], block_idx[1])

        grid_dlm_y, grid_dlm_x = np.meshgrid(idx_dlm, idx_dlm, indexing='xy')
        grid_dnm_y, grid_dnm_x = np.meshgrid(idx_dnm, idx_dnm, indexing='xy')

        self.model_index_dict = {
            'dlm': idx_dlm,
            'dnm': idx_dnm
        }

        self.grid_index_dict = {
            'dlm': (grid_dlm_x, grid_dlm_y),
            'dnm': (grid_dnm_x, grid_dnm_y)
        }

    def _set_parameters_name(self):
        """
        Set parameters names.

        Concatenates the parameter names from DLM and DNM into a single list
        'names_parameters'.
        """
        dlm_names_parameters = self.dlm.names_parameters
        dnm_names_parameters = self.dnm.names_parameters

        dlm_names_parameters.extend(dnm_names_parameters)
        self.names_parameters = dlm_names_parameters

    def _build_F(self, x: np.array = None):
        """
        Build the regression vector.

        Constructs the regression vector 'F' based on the provided regressor
        'x' (only if regression was set) and
        the models DLM and DNM.
        """
        F_dlm = self.dlm._build_F(x=x)
        F_dnm = self.dnm.F

        F = np.vstack((F_dlm, F_dnm))

        return F

    def _build_G(self, x: np.array = None):
        """
        Build the state evolution matrix.

        Constructs the state evolution matrix 'G' based on the
        provided transfer function input 'x'
        (only if transfer function was set) and the models DLM and DNM.
        """
        G_dlm = self.dlm.G
        G_dnm = self.dnm._build_G(x=x)

        G = block_diag(G_dlm, G_dnm)

        return G

    def _build_W(self):
        """
        Build the process noise covariance matrix.

        Constructs the process noise covariance matrix 'W' based on the
        models DLM and DNM.
        """
        W_dlm = self.dlm._build_W()
        W_dnm = self.dnm._build_W()

        W = block_diag(W_dlm, W_dnm)

        return W

    def _build_h(self):
        h_dlm = np.zeros([self.dlm.G.shape[0], 1])
        h_dnm = self.dnm._build_h()

        h = np.vstack([h_dlm, h_dnm])

        return h

    def _calc_prior_mean_and_var(self):
        """
        Calculate prior mean and variance.

        Calculates the prior mean vector 'a' and covariance matrix 'R'
        using the current state and process parameters.
        """
        a = self.G @ self.m + self.h
        P = self.G @ self.C @ self.G.T
        R = (P + self.W)

        return a, R

    def _calc_predictive_mean_and_var(self):
        """
        Calculate predictive mean and variance.

        Calculates the predictive mean 'f' and variance 'q' based on the
        current state and process parameters.
        """
        f = np.ravel(self.F.T @ self.a)[0]
        q = np.ravel(self.F.T @ self.R @ self.F + self.v)[0]
        return f, q

    def _update(self, y: float, X: dict):
        """
        Update the model with new observations.

        Updates the model state and parameters based on the new observation
        'y' and input variables 'X'.
        """
        self.t += 1

        self.F = self._build_F(x=X.get('regression'))
        self.G = self._build_G(x=X.get('transfer_function'))

        self._update_superposition_block_F()
        self._update_superposition_block_G()

        self.W = self._build_W()
        self.h = self._build_h()

        if y is None or np.isnan(y):
            self.m = self.a
            self.C = self.R

            self.a = self.G @ self.m + self.h
            self.R = self.G @ self.C @ self.G.T

            self._update_superposition_block_moments()
        else:
            self.a, self.R = self._calc_prior_mean_and_var()
            self.f, self.q = self._calc_predictive_mean_and_var()

            self.A = (self.R @ self.F) / self.q
            self.e = y - self.f

            self._estimate_observational_variance()

            self._kalman_filter_update()

            self._update_superposition_block_moments()

    def _estimate_observational_variance(self):
        """
        Estimate the observational variance.

        Estimates the observational variance 's' based on the
        model's parameters.
        """
        if self.estimate_V:
            self.r = (self.n + self.e**2 / self.q) / (self.n + 1)
            self.n = self.n + 1
            self.s = self.s * self.r
            self.d = self.s * self.n
        else:
            self.r = 1

    def _kalman_filter_update(self):
        """
        Kalman filter update.

        Updates the space state parameters posterior moments
        using Kalman filter.
        """
        self.a = self.a
        self.R = self.R
        self.m = self.a + self.A * self.e
        self.C = self.r * (self.R - self.q * self.A @ self.A.T)

    def _update_superposition_block_F(self):
        """
        Update the superposition block regression vectors.

        Updates the regression vectors for DLM and DNM based on the
        concatenated observation matrix 'F'.
        """
        idx_dlm = self.model_index_dict.get('dlm')
        idx_dnm = self.model_index_dict.get('dnm')

        self.dlm.F = self.F[idx_dlm]
        self.dnm.F = self.F[idx_dnm]

        self.dlm._update_submodels_F()
        self.dnm._update_submodels_F()

    def _update_superposition_block_G(self):
        """
        Update the superposition block evolution matrices.

        Updates the evolution matrices for DLM and DNM based on the
        concatenated evolution matrix 'G'.
        """
        grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
        grid_dnm_x, grid_dnm_y = self.grid_index_dict.get('dnm')

        self.dlm.G = self.G[grid_dlm_x, grid_dlm_y]
        self.dnm.G = self.G[grid_dnm_x, grid_dnm_y]

        self.dlm._update_submodels_G()
        self.dnm._update_submodels_G()

    def _update_superposition_block_moments(self):
        """
        Update the superposition block moments.

        Updates the posterior mean vectors and covariance matrices for
        DLM and DNM based on the concatenated mean vector 'm' and
        covariance matrix 'C'.
        """
        idx_dlm = self.model_index_dict.get('dlm')
        idx_dnm = self.model_index_dict.get('dnm')

        grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
        grid_dnm_x, grid_dnm_y = self.grid_index_dict.get('dnm')

        self.dlm.m = self.m[idx_dlm]
        self.dnm.m = self.m[idx_dnm]

        self.dlm.C = self.C[grid_dlm_x, grid_dlm_y]
        self.dnm.C = self.C[grid_dnm_x, grid_dnm_y]

        self.dlm.s = self.s
        self.dnm.s = self.s

        if self.dnm.autoregressive_model.order > 0:
            self.v = 0
        else:
            self.v = self.s
            self.dnm.v = self.s

        self.dlm._update_submodels_moments()
        self.dnm._update_submodels_moments()

    def fit(self,
            y: np.ndarray,
            X: dict = {},
            level: float = 0.05,
            smooth: bool = False):
        """
        Fit the model to the data.

        Fits the model to the provided data 'y' with optional regressors 'X'
        and returns the fitted model object.
        """
        # Validate input
        validation.validate_input_dict(mod=self, X=X)

        # Analysis
        foward_dict = _foward_filter(mod=self, y=y, X=X, level=level)
        self.dict_filter = copy(foward_dict.get('filter'))
        self.dict_state_params = copy(foward_dict.get('state_params'))
        self.dict_state_evolution = copy(foward_dict.get('state_evolution'))

        if smooth:
            backward_dict = _backward_smoother(mod=self, X=X, level=level)
            self.dict_smooth = copy(backward_dict.get('smooth'))
            self.dict_smooth_params = copy(backward_dict.get('smooth_params'))

        self.llk = get_predictive_log_likelihood(mod=self)

        return self

    def predict(
            self,
            k: int,
            X: dict = {},
            level: float = 0.05):
        """
        Perform k-step ahead prediction.

        Performs k-step ahead prediction using the fitted model
        and optional regressors 'X' and returns the predicted values and
        confidence intervals.
        """
        # Validate input
        validation.validate_input_dict(mod=self, X=X)

        # Set moments
        copy_mod = copy(self)
        copy_mod.a = copy_mod.m
        copy_mod.R = copy_mod.C

        # Set results dictionary
        dict_state_params = {'a': [], 'R': []}
        dict_kstep_forecast = {'t': [], 'f': [], 'q': []}

        # Set input
        Xt = {'regression': [], 'transfer_function': []}
        copy_X = set_X_dict(mod=copy_mod, nobs=k, X=X)

        # K steps-a-head forecast
        for t in range(k):
            # Prior distribution moments
            Xt['regression'] = copy_X['regression'][t, :]
            Xt['transfer_function'] = copy_X['transfer_function'][t, :, :]

            copy_mod.F = copy_mod._build_F(x=Xt['regression'])
            copy_mod.G = copy_mod._build_G(x=Xt['transfer_function'])

            copy_mod._update_superposition_block_F()
            copy_mod._update_superposition_block_G()

            copy_mod.W = copy_mod._build_W()
            copy_mod.h = copy_mod._build_h()

            copy_mod.a = copy_mod.G @ copy_mod.a + copy_mod.h
            copy_mod.R = copy_mod.G @ copy_mod.R @ copy_mod.G.T + copy_mod.W

            # Predictive Distribution moments
            copy_mod.f, copy_mod.q = copy_mod._calc_predictive_mean_and_var()

            dict_kstep_forecast['t'].append(t+1)
            dict_kstep_forecast['f'].append(copy_mod.f)
            dict_kstep_forecast['q'].append(copy_mod.q)

            dict_state_params['a'].append(copy_mod.a)
            dict_state_params['R'].append(copy_mod.R)

        del copy_mod

        df_predictive = pd.DataFrame(dict_kstep_forecast)

        # Get posterior and predictive dataframes
        df_predictive = _build_predictive_df(
            mod=self, dict_predict=dict_kstep_forecast, level=level)

        df_predict_aR = _build_posterior_df(
            mod=self,
            dict_posterior=dict_state_params,
            entry_m="a",
            entry_v="R",
            t=k,
            level=level)

        # Creat dict of results
        dict_results = {'predictive': df_predictive,
                        'parameters': df_predict_aR}
        return dict_results

    def summary(self):
        """
        Generate a summary of the model.

        Generates a summary of the model including model parameters,
        state estimates, and predictive performance.
        """
        str_summary = summary(mod=self)
        return str_summary
