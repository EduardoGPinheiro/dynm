"""Dynamic Linear Model with transfer function."""
import numpy as np
from dynm.utils.algebra import _build_W_complete


class SeasonalFourier():
    """Class for defining seasonal Fourier model in state space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 seas_period: int = None,
                 seas_harm_components: list = None,
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
        self.nseas = 2 * len(seas_harm_components)
        self.seas_period = seas_period
        self.seas_harm_components = seas_harm_components
        self.discount = discount

        self.m = m0.reshape(-1, 1)
        self.C = C0

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

    def _build_F(self):
        seas_harm_components = self.seas_harm_components

        p = len(seas_harm_components)
        n = 2 * p

        F = np.zeros([n, 1])
        F[0:n:2] = 1

        return F.reshape(-1, 1)

    def _build_G(self):
        seas_period = self.seas_period
        seas_harm_components = self.seas_harm_components

        p = len(seas_harm_components)
        n = 2 * p
        G = np.zeros([n, n])

        for j in range(p):
            c = np.cos(2*np.pi*seas_harm_components[j] / seas_period)
            s = np.sin(2*np.pi*seas_harm_components[j] / seas_period)
            idx = 2*j
            G[idx:(idx+2), idx:(idx+2)] = np.array([[c, s], [-s, c]])

        return G

    def _update_F(self, x: np.array = None):
        F = self.F
        return F

    def _build_P(self):
        return self.G @ self.C @ self.G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W_complete(mod=self, P=P)
        else:
            W = self.W
        return W
