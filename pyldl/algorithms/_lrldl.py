import numpy as np

from pyldl.algorithms.utils import svt
from pyldl.algorithms.base import BaseADMM, BaseLDL

from pyldl.utils import binaryzation


class _LRLDL(BaseADMM, BaseLDL):

    def __init__(self, mode='threshold', param=None, random_state=None):
        super().__init__(random_state)
        self._mode = mode
        self._param = param

    def _update_W(self):
        OTX = self._O.T @ self._X
        temp1 = (self._rho * self._Z.T - self._V.T + self._L.T) @ OTX + self._y.T @ self._X
        temp2 = self._X.T @ self._X + 2 * self._I1 + (1 + self._rho) * self._X.T @ self._O @ OTX
        self._W = np.transpose(temp1 @ np.linalg.inv(temp2))

        WXT = self._W.T @ self._X.T
        temp1 = (1 + self._rho) * WXT.T @ WXT + 2 * self._I2
        temp2 = WXT.T @ (self._L.T + self._rho * self._Z.T - self._V.T)
        self._O = np.linalg.inv(temp1) @ temp2

    def _update_Z(self):
        A = self._W.T @ self._X.T @ self._O + self._V.T / self._rho
        tau = self._alpha / self._rho
        self._Z = np.transpose(svt(A, tau))

    def _update_V(self):
        self._V = np.transpose(self._V.T + self._rho * (self._W.T @ self._X.T @ self._O - self._Z.T))
        self._rho *= 1.1

    def _before_train(self):
        self._O = np.random.random((self._X.shape[0], self._X.shape[0]))
        self._L = binaryzation(self._y, method=self._mode, param=self._param)

        self._I1 = self._beta * np.eye(self._n_features)
        self._I2 = self._beta * np.eye(self._X.shape[0])

    def fit(self, X, y, alpha=1e-3, beta=1e-3, rho=1e-3, **kwargs):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, rho=rho, **kwargs)


class TLRLDL(_LRLDL):

    def __init__(self, param=None, random_state=None):
        super().__init__('threshold', param, random_state)


class TKLRLDL(_LRLDL):

    def __init__(self, param=None, random_state=None):
        super().__init__('topk', param, random_state)
