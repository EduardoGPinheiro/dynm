"""Polynomial model in State Space form."""
import numpy as np
from dynm.utils.algebra import _build_W_diagonal


class Polynomial():
    """Class for defining polynomial model in state space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 ntrend: int,
                 discount: float = .98,
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
        self.ntrend = ntrend
        self.m = m0.reshape(-1, 1)  # Validar entrada de dimensões
        self.C = C0

        self.discount = discount

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

    def _build_F(self):
        ntrend = self.ntrend
        F = np.ones(ntrend)

        if ntrend == 2:
            F[1] = 0

        return F.reshape(-1, 1)

    def _build_G(self):
        ntrend = self.ntrend
        G = np.identity(ntrend)

        if ntrend == 2:
            G[0, 1] = 1

        return G

    def _update_F(self, x: np.array = None):
        F = self.F
        return F

    def _build_P(self):
        return self.G @ self.C @ self.G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W_diagonal(mod=self, P=P)
        else:
            W = self.W
        return W
