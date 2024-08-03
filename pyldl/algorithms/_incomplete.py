import numpy as np

from qpsolvers import solve_qp

from pyldl.algorithms.base import BaseLDL


class IncomLDL(BaseLDL):
    """IncomLDL is proposed in paper Incomplete Label Distribution Learning.
    """

    @staticmethod
    def svt(A, tau):
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        S_thresh = np.maximum(S - tau, 0)
        return U @ np.diag(S_thresh) @ VT

    def _update_Z(self):
        A = self._X @ self._W + self._V / self._rho
        tau = self._alpha / self._rho
        self._Z = self.svt(A, tau)

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

    def _update_V(self):
        self._V = self._V + self._rho * (self._X @ self._W - self._Z)

    def fit(self, X, y, mask, alpha=1e-3, rho=1., max_iterations=100):
        super().fit(X, y)
        self._alpha = alpha
        self._rho = rho
        self._mask = np.where(mask, 0., 1.)
        self._W = np.ones((self._n_features, self._n_outputs))
        self._Z = np.ones((self._X.shape[0], self._n_outputs))
        self._V = np.ones((self._X.shape[0], self._n_outputs))
        for _ in range(max_iterations):
            self._update_W()
            self._update_Z()
            self._update_V()
    
    def predict(self, X):
        return X @ self._W
