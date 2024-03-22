"""Utils functions."""
import numpy as np
import copy
from dynm.utils.algebra import _build_Gnonlinear, _build_W_diagonal


class AutoRegressive():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 order: int,
                 discount: float = None,
                 W: np.ndarray = None,
                 V: float = None):
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
        self.order = order
        self.m = m0.reshape(-1, 1)
        self.C = C0

        if V is None:
            self.n = 1
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.s = V
            self.estimate_V = False

        self.discount = discount
        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

        # Get index for blocks
        block_idx = np.cumsum([order, order])
        self.index_dict = {
            'response': np.arange(0, block_idx[0]),
            'decay': np.arange(block_idx[0], block_idx[1])
        }

    def _build_F(self, x: np.array = None):
        F = np.zeros(2 * self.order)
        F[0] = 1
        return F.reshape(-1, 1)

    def _build_G(self):
        m = self.m
        order = self.order
        G = _build_Gnonlinear(m=m, order=order)
        return G

    def _build_h(self):
        G_ = copy.deepcopy(self.G)
        idx = np.ix_(self.index_dict.get('response'),
                     self.index_dict.get('decay'))

        G_[idx] = G_[idx] * 0.0

        m = self.m.T
        h = (G_ - self.G) @ m.T

        return h

    def _build_P(self):
        return self.G @ self.C @ self.G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W_diagonal(mod=self, P=P)
            W[1:self.order, 1:self.order] = W[1:self.order, 1:self.order] * 0.0
            W[0, 0] = self.s
        else:
            W = self.W
        return W
