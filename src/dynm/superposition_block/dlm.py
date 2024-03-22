"""Linear Structural Block."""
import numpy as np
import copy
from scipy.linalg import block_diag
from dynm.sub_model.polynomial import Polynomial
from dynm.sub_model.regression import Regression
from dynm.sub_model.seasonal_fourier import SeasonalFourier
from dynm.sub_model.nullmodel import NullModel
from dynm.utils import validation


class DynamicLinearModel():
    """Class for definition of dynamic linear model structural block."""

    def __init__(self, model_dict: dict):
        """Define model.

        Define model with system equations components
        and initial information for prior moments.

        Parameters
        ----------
        model_dict : dict
            Dictionary containing prior moments and other model definition \
            parameters for each structural block: polynomial, regression and
            seasonal.

            Each structural block should be identified by its label and \
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

        """
        self.model_dict = copy.deepcopy(model_dict)

        self._validate_model_dict_keys()

        self._validate_model_dict_mean_array()

        self._validate_model_dict_cov_matrix()

        self._validate_model_dict_discount()

        self._set_submodels()

        self._concatenate_regression_vector()

        self._concatenate_evolution_matrix()

        self._concatenate_prior_mean()

        self._concatenate_prior_covariance_matrix()

        self._set_submodels_block_index()

        self._set_parameters_name()

    def _validate_model_dict_keys(self):
        """
        Validate keys in the model dictionary.

        Raises
        ------
        ValueError
            If required keys are missing.
        """
        poly_model_dict = self.model_dict.get('polynomial')
        regn_model_dict = self.model_dict.get('regression')
        seas_model_dict = self.model_dict.get('seasonal')

        if poly_model_dict is not None:
            validation.validate_polynomial_model_dict_keys(
                model_dict=poly_model_dict)

        if regn_model_dict is not None:
            validation.validate_regression_model_dict_keys(
                model_dict=regn_model_dict)

        if seas_model_dict is not None:
            validation.validate_seasonal_model_dict_keys(
                model_dict=seas_model_dict)

    def _validate_model_dict_mean_array(self):
        """
        Validate mean array in the model dictionary.

        Raises
        ------
        ValueError
            If prior mean arrays are incorrect.
        """
        poly_model_dict = self.model_dict.get('polynomial')
        regn_model_dict = self.model_dict.get('regression')
        seas_model_dict = self.model_dict.get('seasonal')

        if self.model_dict.get('polynomial') is not None:
            validation.validate_model_dict_polynomial_mean_array(
                model_dict=poly_model_dict)

        if self.model_dict.get('regression') is not None:
            validation.validate_model_dict_regression_mean_array(
                model_dict=regn_model_dict)

        if self.model_dict.get('seasonal') is not None:
            validation.validate_model_dict_seasonal_mean_array(
                model_dict=seas_model_dict)

    def _validate_model_dict_cov_matrix(self):
        """
        Validate covariance matrix in the model dictionary.

        Raises
        ------
        ValueError
            If prior covariance matrices are incorrect.
        """
        poly_model_dict = self.model_dict.get('polynomial')
        regn_model_dict = self.model_dict.get('regression')
        seas_model_dict = self.model_dict.get('seasonal')

        if self.model_dict.get('polynomial') is not None:
            validation.validate_model_dict_polynomial_covariance_matrix(
                model_dict=poly_model_dict)

        if self.model_dict.get('regression') is not None:
            validation.validate_model_dict_regression_covariance_matrix(
                model_dict=regn_model_dict)

        if self.model_dict.get('seasonal') is not None:
            validation.validate_model_dict_seasonal_covariance_matrix(
                model_dict=seas_model_dict)

    def _validate_model_dict_discount(self):
        """
        Validate discount in the model dictionary.

        Raises
        ------
        ValueError
            If the discount factor is not a scalar or falls
            outside the [0, 1] interval.
        """
        poly_model_dict = self.model_dict.get('polynomial')
        regn_model_dict = self.model_dict.get('regression')
        seas_model_dict = self.model_dict.get('seasonal')

        if self.model_dict.get('polynomial') is not None:
            validation.validate_model_dict_polynomial_discount_array(
                model_dict=poly_model_dict)

        if self.model_dict.get('regression') is not None:
            validation.validate_model_dict_regression_discount_array(
                model_dict=regn_model_dict)

        if self.model_dict.get('seasonal') is not None:
            validation.validate_model_dict_seasonal_discount_array(
                model_dict=seas_model_dict)

    def _set_submodels(self):
        """Set submodels based on the model dictionary."""
        if self.model_dict.get('polynomial') is not None:
            submod_dict = self.model_dict.get('polynomial')
            self.ntrend = submod_dict.get('ntrend')

            polynomial_mod = Polynomial(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                ntrend=submod_dict.get('ntrend'),
                W=submod_dict.get('W'))
        else:
            polynomial_mod = NullModel()

        if self.model_dict.get('regression') is not None:
            submod_dict = self.model_dict.get('regression')
            self.nregn = submod_dict.get('nregn')

            regression_mod = Regression(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                nregn=submod_dict.get('nregn'),
                W=submod_dict.get('W'))

        else:
            regression_mod = NullModel()

        if self.model_dict.get('seasonal') is not None:
            submod_dict = self.model_dict.get('seasonal')

            self.nseas = 2 * len(submod_dict.get('seas_harm_components'))
            self.seas_period = submod_dict.get('seas_period')
            self.seas_harm_components = submod_dict.get('seas_harm_components')

            seasonal_mod = SeasonalFourier(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                seas_period=submod_dict.get('seas_period'),
                seas_harm_components=submod_dict.get('seas_harm_components'),
                W=submod_dict.get('W'))

        else:
            seasonal_mod = NullModel()

        self.polynomial_model = polynomial_mod
        self.regression_model = regression_mod
        self.seasonal_model = seasonal_mod

    def _concatenate_regression_vector(self):
        """Concatenate regression vectors."""
        self.F = np.vstack((self.polynomial_model.F,
                            self.regression_model.F,
                            self.seasonal_model.F))

    def _concatenate_evolution_matrix(self):
        """Concatenate equation evolution matrices."""
        self.G = block_diag(self.polynomial_model.G,
                            self.regression_model.G,
                            self.seasonal_model.G)

    def _concatenate_prior_mean(self):
        """Concatenate prior means vectors."""
        self.a = np.vstack((self.polynomial_model.m,
                            self.regression_model.m,
                            self.seasonal_model.m))
        self.m = np.vstack((self.polynomial_model.m,
                            self.regression_model.m,
                            self.seasonal_model.m))

    def _concatenate_prior_covariance_matrix(self):
        """Concatenate prior covariance matrices."""
        self.R = block_diag(self.polynomial_model.C,
                            self.regression_model.C,
                            self.seasonal_model.C)
        self.C = block_diag(self.polynomial_model.C,
                            self.regression_model.C,
                            self.seasonal_model.C)

    def _set_submodels_block_index(self):
        """Set block indices for submodels."""
        nparams_polynomial = len(self.polynomial_model.m)
        nparams_regression = len(self.regression_model.m)
        nparams_seasonal = len(self.seasonal_model.m)

        block_idx = np.cumsum([nparams_polynomial,
                               nparams_regression,
                               nparams_seasonal])

        idx_poly = np.arange(0, block_idx[0])
        idx_regn = np.arange(block_idx[0], block_idx[1])
        idx_seas = np.arange(block_idx[1], block_idx[2])

        grid_poly_y, grid_poly_x = np.meshgrid(
            idx_poly, idx_poly, indexing='xy')
        grid_regn_y, grid_regn_x = np.meshgrid(
            idx_regn, idx_regn, indexing='xy')
        grid_seas_y, grid_seas_x = np.meshgrid(
            idx_seas, idx_seas, indexing='xy')

        self.model_index_dict = {
            'polynomial': idx_poly,
            'regression': idx_regn,
            'seasonal': idx_seas
        }

        self.grid_index_dict = {
            'polynomial': (grid_poly_x, grid_poly_y),
            'regression': (grid_regn_x, grid_regn_y),
            'seasonal': (grid_seas_x, grid_seas_y)
        }

    def _set_parameters_name(self):
        """Set parameter names."""
        level_labels = \
            ['intercept_' + str(i+1)
             for i in range(self.polynomial_model.ntrend)]

        regn_labels = \
            ['beta_' + str(i+1)
             for i in range(self.regression_model.nregn)]

        seas_labels = \
            ['seas_harm_' + str(i+1)
             for i in range(self.seasonal_model.nseas)]

        names_parameters = (level_labels + regn_labels + seas_labels)
        self.names_parameters = names_parameters

    def _build_F(self, x: np.array = None):
        """Build regression vector."""
        F_poly = self.polynomial_model.F
        F_regn = self.regression_model._update_F(x=x)
        F_seas = self.seasonal_model.F

        F = np.vstack([F_poly, F_regn, F_seas])

        return F

    def _build_G(self):
        """Build system equation evolution matrix."""
        G_poly = self.polynomial_model.G
        G_regn = self.regression_model.G
        G_seas = self.seasonal_model.G

        G = block_diag(G_poly, G_regn, G_seas)

        return G

    def _build_W(self):
        """Build process noise covariance matrix."""
        P_poly = self.polynomial_model._build_P()
        P_regn = self.regression_model._build_P()
        P_seas = self.seasonal_model._build_P()

        W_poly = self.polynomial_model._build_W(P=P_poly)
        W_regn = self.regression_model._build_W(P=P_regn)
        W_seas = self.seasonal_model._build_W(P=P_seas)

        W = block_diag(W_poly, W_regn, W_seas)

        return W

    def _update_submodels_F(self):
        """Update regression vector for each submodel."""
        idx_poly = self.model_index_dict.get('polynomial')
        idx_regn = self.model_index_dict.get('regression')
        idx_seas = self.model_index_dict.get('seasonal')

        self.polynomial_model.F = self.F[idx_poly]
        self.regression_model.F = self.F[idx_regn]
        self.seasonal_model.F = self.F[idx_seas]

    def _update_submodels_G(self):
        """Update system equation evolution matrices for each submodel."""
        grid_poly_x, grid_poly_y = self.grid_index_dict.get('polynomial')
        grid_regn_x, grid_regn_y = self.grid_index_dict.get('regression')
        grid_seas_x, grid_seas_y = self.grid_index_dict.get('seasonal')

        self.polynomial_model.G = self.G[grid_poly_x, grid_poly_y]
        self.regression_model.G = self.G[grid_regn_x, grid_regn_y]
        self.seasonal_model.G = self.G[grid_seas_x, grid_seas_y]

    def _update_submodels_moments(self):
        """Update prior moments for each submodel."""
        idx_poly = self.model_index_dict.get('polynomial')
        idx_regn = self.model_index_dict.get('regression')
        idx_seas = self.model_index_dict.get('seasonal')

        grid_poly_x, grid_poly_y = self.grid_index_dict.get('polynomial')
        grid_regn_x, grid_regn_y = self.grid_index_dict.get('regression')
        grid_seas_x, grid_seas_y = self.grid_index_dict.get('seasonal')

        self.polynomial_model.m = self.m[idx_poly]
        self.regression_model.m = self.m[idx_regn]
        self.seasonal_model.m = self.m[idx_seas]

        self.polynomial_model.C = self.C[grid_poly_x, grid_poly_y]
        self.regression_model.C = self.C[grid_regn_x, grid_regn_y]
        self.seasonal_model.C = self.C[grid_seas_x, grid_seas_y]
