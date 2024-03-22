"""Regression in State Space form."""
import numpy as np
from dynm.utils.algebra import _build_W_complete


class Regression():
    """Class for defining seasonal regression model in state space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 nregn: int,
                 discount: float = .998,
                 W: np.ndarray = None):
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
        self.nregn = nregn
        self.discount = discount

        self.m = m0.reshape(-1, 1)
        self.C = C0

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F(x=0)
        self.G = self._build_G()

    def _build_F(self, x: np.array):
        nregn = self.nregn
        F = np.ones(nregn) * x
        return F.reshape(-1, 1)

    def _build_G(self):
        nregn = self.nregn
        G = np.identity(nregn)
        return G

    def _update_F(self, x: np.array = None):
        F = self.F
        F[:, 0] = np.ravel(x)
        return F

    def _build_P(self):
        return self.G @ self.C @ self.G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W_complete(mod=self, P=P)
        else:
            W = self.W
        return W
