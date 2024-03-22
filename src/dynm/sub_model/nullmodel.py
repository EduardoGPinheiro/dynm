"""Class for null model."""
import numpy as np


class NullModel():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self):
        """Define model.

        Define model with observation/system equations components \
        and initial information for prior moments.

        Parameters
        ----------
        m0 : np.ndarray
            prior mean for state space components.
        C0 : np.ndarray
            prior covariance for state space components.
        delta : float
            discount factor.

        """
        self.order = 0
        self.lambda_order = 0
        self.gamma_order = 0
        self.ntrend = 0
        self.nregn = 0
        self.nseas = 0
        self.ntfm = 0
        self.m = np.array([np.array([])]).reshape(-1, 1)
        self.C = np.empty([0, 0])

        self.F = self._build_F()
        self.G = self._build_G()

        # Get index for blocks
        self.index_dict = {
            'response': np.arange(0, 0),
            'decay': np.arange(0, 0)
        }

    def _update_F(self, x: np.array = None):
        return np.empty([0, 0]).reshape(-1, 1)

    def _build_F(self):
        return np.empty([0, 0]).reshape(-1, 1)

    def _build_G(self, x: float = None):
        return np.empty([0, 0])

    def _build_h(self):
        return np.empty([0, 0]).reshape(-1, 1)

    def _build_P(self, G: np.array = None):
        return np.empty([0, 0])

    def _build_W(self, P: np.array):
        return np.empty([0, 0])
