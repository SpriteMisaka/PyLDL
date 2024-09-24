import numpy as np

from qpsolvers import solve_qp

from pyldl.algorithms.utils import svt, proj
from pyldl.algorithms.base import BaseADMM, BaseIncomLDL


class IncomLDL(BaseADMM, BaseIncomLDL):
    """:class:`IncomLDL <pyldl.algorithms.IncomLDL>` is proposed in paper :cite:`2017:xu`.
    """

    def _update_W(self):

        G = -np.eye(self._n_outputs, dtype=np.float64)
        h = np.zeros((self._n_outputs, 1), dtype=np.float64)
        A = np.ones((1, self._n_outputs), dtype=np.float64)
        b = np.array([1.], dtype=np.float64)

        M = np.zeros_like(self._y)

        for i in range(self._X.shape[0]):
            P = np.diag((1 + self._rho) * self._mask[i] + self._rho * (1 - self._mask[i])).astype(np.float64)
            ql = (self._V[i] - self._y[i] * self._mask[i] - self._rho * self._Z[i]) * self._mask[i]
            qr = (self._V[i] - self._rho * self._Z[i]) * (1 - self._mask[i])
            q = np.transpose(ql + qr).astype(np.float64)
            M[i] = solve_qp(P, q, G, h, A, b, solver='quadprog')

        self._W = np.linalg.pinv(np.transpose(self._X) @ self._X) @ np.transpose(self._X) @ M

    def _update_Z(self):
        A = self._X @ self._W + self._V / self._rho
        tau = self._alpha / self._rho
        self._Z = svt(A, tau)

    def fit(self, X, y, mask, alpha=1e-3, **kwargs):
        self._alpha = alpha
        return super().fit(X, y, mask=mask, **kwargs)


class WInLDL(BaseADMM, BaseIncomLDL):
    """:class:`WInLDL <pyldl.algorithms.WInLDL>` is proposed in paper :cite:`2024:li`.
    """

    def _update_W(self):
        self._W = np.linalg.solve(
            np.transpose(self._X) @ self._X + 1e-5 * np.eye(self._n_features),
            np.transpose(self._X) @ (self._Z - self._V / self._rho)
        )

    def _update_Z(self):
        self._update_Q()
        Y = (self._X @ self._W) * (1 - self._mask) + self._y
        numerator = self._rho * self._X @ self._W + self._V + self._Q * self._Q * Y
        denominator = self._Q * self._Q + self._rho
        self._Z = proj(numerator / denominator)

    def _update_Q(self):
        a = 1 + self._current_iteration / self._max_iterations
        self._Q2 = np.power(a, np.tile(self._avg, (self._y.shape[0], 1))) * (1 - self._mask)
        self._Q = self._Q1 + self._Q2

    def _before_train(self):
        self._avg = np.sum(self._y, axis=0) / np.count_nonzero(self._y, axis=0)
        self._Q1 = np.exp2(1 - self._y) * self._mask

    def fit(self, X, y, mask, rho=2., **kwargs):
        return super().fit(X, y, mask=mask, rho=rho, **kwargs)
