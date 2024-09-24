import numpy as np
from scipy.optimize import minimize

from pyldl.algorithms.base import BaseADMM, BaseLDL
from pyldl.algorithms.utils import soft_thresholding, solvel21


class LDLSF(BaseADMM, BaseLDL):
    """:class:`LDLSF <pyldl.algorithms.LDLSF>` is proposed in paper :cite:`2019:ren`.

    :term:`ADMM` is used as optimization algorithm.
    """

    def _update_W(self):

        def _obj_func(w):
            self._W = w.reshape(self._n_features, self._n_outputs)
            XW = self._X @ self._W
            s = (np.sum(XW, axis=1) - 1).reshape(-1, 1)
            lT = np.ones((1, self._n_outputs))
            W12 = self._W - self._W1 - self._W2

            def _W_loss():
                fro = np.linalg.norm(XW - self._y, 'fro') ** 2 / 2.
                con1 = np.linalg.norm(W12, 'fro') ** 2
                con2 = np.linalg.norm(s) ** 2
                con = self._rho * (con1 + con2) / 2.
                inn1 = np.sum(self._V * (W12))
                inn2 = np.sum(self._V2 * s)
                inn = inn1 + inn2
                tr = np.trace(XW @ self._PR @ XW.T)
                return fro + con + inn + self._gamma * tr

            def _W_grad():
                grad = self._X.T @ (XW - self._y)
                grad += self._V
                grad += self._rho * (W12)
                grad += self._X.T @ self._V2 @ lT
                grad += self._X.T @ (self._rho * s) @ lT
                grad += self._gamma * self._X.T @ XW @ (self._PR + self._PR.T)
                return grad.reshape(-1, )

            return _W_loss(), _W_grad()

        w0 = self._W.reshape(-1, ).copy()
        optimize_result = minimize(_obj_func, w0, method='L-BFGS-B', jac=True)
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        self._W1 = soft_thresholding(self._W - self._W2 + self._V / self._rho, self._alpha / self._rho)
        self._W2 = solvel21(self._W - self._W1 + self._V / self._rho, self._beta / self._rho)

    def _update_V(self):
        XW = self._X @ self._W
        s = (np.sum(XW, axis=1) - 1).reshape(-1, 1)
        self._V = self._V + self._rho * (self._W - self._W1 - self._W2)
        self._V2 = self._V2 + self._rho * s

    def _before_train(self):
        self._W1 = .5 * np.eye(self._n_features, self._n_outputs)
        self._W2 = .5 * np.eye(self._n_features, self._n_outputs)
        self._V2 = np.zeros((self._X.shape[0], 1))
        R = np.corrcoef(self._y, rowvar=False)
        P = np.diag(R @ np.ones((self._n_outputs, )))
        self._PR = P - R

    def _get_default_model(self):
        _W = np.eye(self._n_features, self._n_outputs)
        _V = np.zeros((self._n_features, self._n_outputs))
        return _W, None, _V

    @property
    def constraint(self):
        XW = self._X @ self._W
        return [[self._W1 + self._W2, self._W],
                [np.ones((self._X.shape[0], 1)), XW @ np.ones((self._n_outputs, 1))]]

    @property
    def params(self):
        return [self._W, self._W1, self._W2]

    @property
    def Vs(self):
        return [self._V, self._V2]

    def fit(self, X, y, alpha=1e-4, beta=1e-2, gamma=1e-3, rho=1e-3, **kwargs):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        return super().fit(X, y, rho=rho, **kwargs)
