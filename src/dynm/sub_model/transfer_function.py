"""Utils functions."""
import numpy as np
import copy
from dynm.utils.algebra import _build_Gnonlinear, _build_W_diagonal
from scipy.linalg import block_diag


class TransferFunction():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 gamma_order: int,
                 lambda_order: int,
                 ntfm: int,
                 discount: np.ndarray = None,
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
        self.gamma_order = gamma_order
        self.lambda_order = lambda_order
        self.ntfm = ntfm
        self.m = m0.reshape(-1, 1)
        self.C = C0

        self.discount = discount

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        # Get index for blocks
        self.index_dict = {}

        for n in range(ntfm):
            block_idx = n * (2 * lambda_order + gamma_order) + \
                np.cumsum([lambda_order, lambda_order, gamma_order])
            self.index_dict[n] = {
                'all': np.arange(n * (2 * lambda_order + gamma_order),
                                 block_idx[2]),
                'response': np.arange(n * (2 * lambda_order + gamma_order),
                                      block_idx[0]),
                'decay': np.arange(block_idx[0], block_idx[1]),
                'pulse': np.arange(block_idx[1], block_idx[2])}

        # Build F and G
        self.F = self._build_F()
        self.G = self._build_G(x=np.zeros([ntfm, self.gamma_order]))

    def _build_F(self):
        F = np.array([])

        for i in range(self.ntfm):
            Fi = np.zeros(2 * self.lambda_order + self.gamma_order)
            Fi[0] = 1

            F = np.hstack((F, Fi))

        return F.reshape(-1, 1)

    def _build_G(self, x: np.array):
        m = self.m
        lambda_order = self.lambda_order
        ntfm = self.ntfm

        G = np.empty([0, 0])
        for n in range(ntfm):
            idx_ = np.concatenate((
                self.index_dict.get(n).get('response'),
                self.index_dict.get(n).get('decay'),
                self.index_dict.get(n).get('pulse')))

            m_ = m[idx_]
            Gi = _build_Gnonlinear(m=m_.reshape(-1, 1),
                                   order=lambda_order)

            Hn = np.zeros([Gi.shape[0], self.gamma_order])
            for o in range(self.gamma_order):
                xn = np.ravel(x[n, o])
                Hn[0, o] = xn

            In = np.identity(self.gamma_order)
            Gn = np.block([[Gi, Hn], [Hn.T * 0, In]])
            G = block_diag(G, Gn)

        return G

    def _build_h(self):
        ntfm = self.ntfm
        G_ = copy.deepcopy(self.G)

        for n in range(ntfm):
            idx = np.ix_(self.index_dict.get(n).get('response'),
                         self.index_dict.get(n).get('decay'))
            G_[idx] = G_[idx] * 0.0

        m = self.m.T
        h = (G_ - self.G) @ m.T

        return h

    def _build_P(self):
        return self.G @ self.C @ self.G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W_diagonal(mod=self, P=P)

            for n in range(self.ntfm):
                idx = np.ix_(self.index_dict.get(n).get('response')[1:],
                             self.index_dict.get(n).get('response')[1:])

                W[idx] = W[idx] * 0.0
        else:
            W = self.W
        return W
