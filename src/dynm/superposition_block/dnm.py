"""Non Linear Structural Block."""
import numpy as np
import copy
from dynm.sub_model.nullmodel import NullModel
from dynm.sub_model.autoregressive import AutoRegressive
from dynm.sub_model.transfer_function import TransferFunction
from scipy.linalg import block_diag
from dynm.utils import validation


class DynamicNonLinearModel():
    """Class for definition of dynamic non-linear model structural block."""

    def __init__(self, model_dict: dict, V: float = None):
        """Define model.

        Define model with system equations components \
        and initial information for prior moments.

        Parameters
        ----------
        model_dict : dict
            Dictionary containing prior moments and other model definition \
            parameters for each structural block: transfer function and \
            autoregressive.

            Each structural block should be identified by its label and \
            should have the following parameters:

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
        self.model_dict = copy.deepcopy(model_dict)
        self.V = V

        self._validate_model_dict_keys()

        self._validate_model_dict_mean_array()

        self._validate_model_dict_cov_matrix()

        self._validate_model_dict_discount()

        self._set_submodels()

        self._set_gamma_distribution_parameters()

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
        tf_model_dict = self.model_dict.get('transfer_function')
        ar_model_dict = self.model_dict.get('autoregressive')

        if tf_model_dict is not None:
            validation.validate_transfer_function_model_dict_keys(
                model_dict=tf_model_dict)

        if ar_model_dict is not None:
            validation.validate_autoregressive_model_dict_keys(
                model_dict=ar_model_dict)

    def _validate_model_dict_mean_array(self):
        """
        Validate mean array in the model dictionary.

        Raises
        ------
        ValueError
            If prior mean arrays are incorrect.
        """
        tf_model_dict = self.model_dict.get('transfer_function')
        ar_model_dict = self.model_dict.get('autoregressive')

        if tf_model_dict is not None:
            validation.validate_transfer_function_model_dict_keys(
                model_dict=tf_model_dict)

        if ar_model_dict is not None:
            validation.validate_autoregressive_model_dict_keys(
                model_dict=ar_model_dict)

    def _validate_model_dict_cov_matrix(self):
        """
        Validate covariance matrix in the model dictionary.

        Raises
        ------
        ValueError
            If prior covariance matrices are incorrect.
        """
        tf_model_dict = self.model_dict.get('transfer_function')
        ar_model_dict = self.model_dict.get('autoregressive')

        if tf_model_dict is not None:
            validation.validate_model_dict_transfer_function_covariance_matrix(
                model_dict=tf_model_dict)

        if ar_model_dict is not None:
            validation.validate_model_dict_autoregressive_covariance_matrix(
                model_dict=ar_model_dict)

    def _validate_model_dict_discount(self):
        """
        Validate discount in the model dictionary.

        Raises
        ------
        ValueError
            If the discount array is incompatible with the model
            parameters, or if any element in the discount array falls outside
            the [0, 1] interval.
        """
        tf_model_dict = self.model_dict.get('transfer_function')
        ar_model_dict = self.model_dict.get('autoregressive')

        if tf_model_dict is not None:
            validation.validate_model_dict_transfer_function_discount_array(
                model_dict=tf_model_dict)

        if ar_model_dict is not None:
            validation.validate_model_dict_autoregressive_discount_array(
                model_dict=ar_model_dict)

    def _set_submodels(self):
        """Set submodels based on the model dictionary."""
        if self.model_dict.get('autoregressive') is not None:
            submod_dict = self.model_dict.get('autoregressive')
            autoregressive = AutoRegressive(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                order=submod_dict.get('order'),
                W=submod_dict.get('W'))
        else:
            autoregressive = NullModel()

        if self.model_dict.get('transfer_function') is not None:
            submod_dict = self.model_dict.get('transfer_function')
            transfer_function = TransferFunction(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                lambda_order=submod_dict.get('lambda_order'),
                gamma_order=submod_dict.get('gamma_order'),
                ntfm=submod_dict.get('ntfm'),
                W=submod_dict.get('W'))
        else:
            transfer_function = NullModel()

        self.autoregressive_model = autoregressive
        self.transfer_function_model = transfer_function

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

        if self.autoregressive_model.order > 0:
            self.v = 0
        else:
            self.v = self.s

    def _concatenate_regression_vector(self):
        """Concatenate regression vectors."""
        self.F = np.vstack((self.autoregressive_model.F,
                            self.transfer_function_model.F))

    def _concatenate_evolution_matrix(self):
        """Concatenate equation evolution matrices."""
        self.G = block_diag(self.autoregressive_model.G,
                            self.transfer_function_model.G)

    def _concatenate_prior_mean(self):
        """Concatenate prior means vectors."""
        self.a = np.vstack((self.autoregressive_model.m,
                            self.transfer_function_model.m))
        self.m = np.vstack((self.autoregressive_model.m,
                            self.transfer_function_model.m))

    def _concatenate_prior_covariance_matrix(self):
        """Concatenate prior covariance matrices."""
        self.R = block_diag(self.autoregressive_model.C,
                            self.transfer_function_model.C)
        self.C = block_diag(self.autoregressive_model.C,
                            self.transfer_function_model.C)

    def _set_submodels_block_index(self):
        """Set block indices for submodels."""
        nparams_autoregressive = len(self.autoregressive_model.m)
        nparams_transfer_function = len(self.transfer_function_model.m)

        block_idx = np.cumsum([nparams_autoregressive,
                               nparams_transfer_function])

        idx_ar = np.arange(0, block_idx[0])
        idx_tf = np.arange(block_idx[0], block_idx[1])

        grid_ar_y, grid_ar_x = np.meshgrid(idx_ar, idx_ar, indexing='xy')
        grid_tf_y, grid_tf_x = np.meshgrid(idx_tf, idx_tf, indexing='xy')

        self.model_index_dict = {
            'autoregressive': idx_ar,
            'transfer_function': idx_tf
        }

        self.grid_index_dict = {
            'autoregressive': (grid_ar_x, grid_ar_y),
            'transfer_function': (grid_tf_x, grid_tf_y)
        }

    def _set_parameters_name(self):
        """Set parameter names."""
        ar__response_labels = \
            ['xi_' + str(i+1)
             for i in range(self.autoregressive_model.order)]

        ar__decay_labels = \
            ['phi_' + str(i+1)
             for i in range(self.autoregressive_model.order)]

        tf__response_labels = \
            ['E_' + str(i+1)
             for i in range(self.transfer_function_model.lambda_order)]

        tf__decay_labels = \
            ['lambda_' + str(i+1)
             for i in range(self.transfer_function_model.lambda_order)]

        pulse_labels = \
            ['gamma_' + str(i+1)
             for i in range(self.transfer_function_model.gamma_order)]

        names_parameters = (
            ar__response_labels +
            ar__decay_labels +
            self.transfer_function_model.ntfm *
            (tf__response_labels + tf__decay_labels + pulse_labels))

        self.names_parameters = names_parameters

    def _build_F(self):
        """Build regression vector."""
        F = np.vstack((
            self.autoregressive_model.F,
            self.transfer_function_model.F))

        return F

    def _build_G(self, x: np.array):
        """Build system equation evolution matrix."""
        G_ar = self.autoregressive_model._build_G()
        G_tf = self.transfer_function_model._build_G(x=x)

        G = block_diag(G_ar, G_tf)

        return G

    def _build_W(self):
        """Build process noise covariance matrix."""
        P_ar = self.autoregressive_model._build_P()
        P_tf = self.transfer_function_model._build_P()

        W_ar = self.autoregressive_model._build_W(P=P_ar)
        W_tf = self.transfer_function_model._build_W(P=P_tf)

        W = block_diag(W_ar, W_tf)

        return W

    def _build_h(self):
        h_ar = self.autoregressive_model._build_h()
        h_tf = self.transfer_function_model._build_h()

        h = np.vstack([h_ar, h_tf])

        return h

    def _update_submodels_F(self):
        """Update regression vector for each submodel."""
        idx_ar = self.model_index_dict.get('autoregressive')
        idx_tf = self.model_index_dict.get('transfer_function')

        self.autoregressive_model.F = self.F[idx_ar]
        self.transfer_function_model.F = self.F[idx_tf]

    def _update_submodels_G(self):
        """Update system equation evolution matrices for each submodel."""
        grid_ar_x, grid_ar_y = self.grid_index_dict.get('autoregressive')
        grid_tf_x, grid_tf_y = self.grid_index_dict.get('transfer_function')

        self.autoregressive_model.G = self.G[grid_ar_x, grid_ar_y]
        self.transfer_function_model.G = self.G[grid_tf_x, grid_tf_y]

    def _update_submodels_moments(self):
        """Update prior moments for each submodel."""
        idx_ar = self.model_index_dict.get('autoregressive')
        idx_tf = self.model_index_dict.get('transfer_function')

        grid_ar_x, grid_ar_y = self.grid_index_dict.get('autoregressive')
        grid_tf_x, grid_tf_y = self.grid_index_dict.get('transfer_function')

        self.autoregressive_model.m = self.m[idx_ar]
        self.transfer_function_model.m = self.m[idx_tf]

        self.autoregressive_model.C = self.C[grid_ar_x, grid_ar_y]
        self.transfer_function_model.C = self.C[grid_tf_x, grid_tf_y]

        self.autoregressive_model.s = self.s
        self.transfer_function_model.s = self.s

        if self.autoregressive_model.order > 0:
            self.v = 0
        else:
            self.v = self.s
