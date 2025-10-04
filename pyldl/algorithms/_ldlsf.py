import numpy as np

from numba import jit

from pyldl.algorithms.base import BaseADMM, BaseLDL


@jit(nopython=True)
def _update_W_numba(X, D, W, W1, W2, V, V2, PR, gamma, rho):
    XW = X @ W
    W12 = W - W1 - W2
    s = (np.sum(XW, axis=1) - 1).reshape(-1, 1)
    fro = np.linalg.norm(XW - D) ** 2 / 2.
    con1 = np.linalg.norm(W12) ** 2
    con2 = np.linalg.norm(s) ** 2
    con = rho * (con1 + con2) / 2.
    inn1 = np.sum(V * W12)
    inn2 = np.sum(V2 * s)
    inn = inn1 + inn2
    tr = np.trace(XW @ PR @ XW.T)
    loss = fro + con + inn + gamma * tr
    lT = np.ones((1, D.shape[1]))
    grad = X.T @ (XW - D)
    grad += V
    grad += rho * W12
    grad += X.T @ (V2 + rho * s) @ lT
    grad += gamma * X.T @ XW @ (PR + PR.T)
    return loss, grad.reshape(-1, )


@jit(nopython=True)
def _update_V_numba(X, W, W1, W2, V, V2, rho):
    s = (np.sum(X @ W, axis=1) - 1).reshape(-1, 1)
    return V + rho * (W - W1 - W2), V2 + rho * s


@jit(nopython=True)
def _calculate_PR_numba(D):
    R = np.corrcoef(D, rowvar=False)
    P = np.diag(R @ np.ones((D.shape[1], )))
    return P - R


class LDLSF(BaseADMM, BaseLDL):
    """:class:`LDLSF <pyldl.algorithms.LDLSF>` is proposed in paper :cite:`2019:ren`. 
    SF refers to *specific features*.

    :term:`ADMM` is used as optimization algorithm.
    """

    def __init__(self, alpha=1e-4, beta=1e-2, gamma=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _update_W(self):
        from scipy.optimize import minimize
        from pyldl.algorithms.utils import soft_thresholding, solvel21

        def _obj_func(w):
            self._W = w.reshape(self._n_features, self._n_outputs)
            return _update_W_numba(self._X, self._D, self._W, self._W1, self._W2,
                                   self._V, self._V2, self._PR, self.gamma, self._rho)

        optimize_result = minimize(_obj_func, self._W.reshape(-1, ), method='L-BFGS-B', jac=True)
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        self._W1 = soft_thresholding(self._W - self._W2 + self._V / self._rho, self.alpha / self._rho)
        self._W2 = solvel21(self._W - self._W1 + self._V / self._rho, self.beta / self._rho)

    def _update_V(self):
        self._V, self._V2 = _update_V_numba(self._X, self._W, self._W1, self._W2,
                                            self._V, self._V2, self._rho)

    def _before_train(self):
        self._W1 = .5 * np.eye(self._n_features, self._n_outputs)
        self._W2 = .5 * np.eye(self._n_features, self._n_outputs)
        self._V2 = np.zeros((self._n_samples, 1))
        self._PR = _calculate_PR_numba(self._D)

    def _get_default_model(self):
        _W = np.eye(self._n_features, self._n_outputs)
        _V = np.zeros((self._n_features, self._n_outputs))
        return _W, None, _V

    @property
    def constraint(self):
        XW = self._X @ self._W
        return [[self._W1 + self._W2, self._W],
                [np.ones((self._n_samples, 1)), XW @ np.ones((self._n_outputs, 1))]]

    @property
    def params(self):
        return [self._W, self._W1, self._W2]

    @property
    def Vs(self):
        return [self._V, self._V2]

    def fit(self, X, D, rho=1e-3, **kwargs):
        return super().fit(X, D, rho=rho, **kwargs)
