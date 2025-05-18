import numpy as np

from numba import jit

from scipy.optimize import minimize

from sklearn.cluster import KMeans

from pyldl.algorithms.base import BaseADMM, BaseLDL
from pyldl.algorithms.utils import svt, solvel21, pairwise_euclidean


@jit(nopython=True)
def _get_D_pred(X, W):
    XW = X @ W
    return np.exp(XW) / np.sum(np.exp(XW), axis=1).reshape(-1, 1)


@jit(nopython=True)
def _get_D_pred_DSE(D, S, E, X, W):
    D_pred = _get_D_pred(X, W)
    return D_pred, D - D_pred @ S - E


@jit(nopython=True)
def _update_W_numba(X, D, W, S, E, V, alpha, rho):
    D_pred, DSE = _get_D_pred_DSE(D, S, E, X, W)
    DD2 = D_pred - D_pred ** 2
    kl = np.sum(D * (np.log(D) - np.log(D_pred)))
    inn = np.sum(V * DSE)
    fro1 = np.linalg.norm(W) ** 2
    fro2 = rho * np.linalg.norm(DSE) ** 2 / 2.
    loss = kl + inn + alpha * fro1 + fro2
    grad = X.T @ (D_pred - D)
    grad += 2 * alpha * W
    grad -= X.T @ (DD2 * V) @ S.T
    grad -= rho * X.T @ (DD2 * DSE) @ S.T
    return loss, grad.reshape(-1, )


@jit(nopython=True)
def _update_S_numba(X, D, W, S, E, Z, V, V2, P, sumP, n_clusters, delta, rho):
    D_pred, DSE = _get_D_pred_DSE(D, S, E, X, W)
    inn1 = np.sum(V * DSE)
    inn2 = np.sum(V2 * (S - Z))
    inn = inn1 + inn2
    fro1 = np.linalg.norm(S - Z) ** 2
    fro2 = np.linalg.norm(DSE) ** 2
    fro = rho * (fro1 + fro2) / 2.
    pairwise = 0.
    for i in range(n_clusters):
        pairwise -= np.sum(S * P[i])
    loss = inn + fro + delta * pairwise
    grad = - V.T @ D_pred + V2.T
    grad += rho * (S - Z - D_pred.T @ DSE)
    grad -= sumP
    return loss, grad.reshape(-1, )


@jit(nopython=True)
def _update_V_numba(X, D, W, S, E, Z, V, V2, rho):
    _, DSE = _get_D_pred_DSE(D, S, E, X, W)
    return V + rho * DSE, V2 + rho * (S - Z)


class LDL_LCLR(BaseADMM, BaseLDL):
    """:class:`LDL-LCLR <pyldl.algorithms.LDL_LCLR>` is proposed in paper :cite:`2019:ren2`.

    :term:`ADMM` is used as the optimization algorithm.
    """

    def __init__(self, n_clusters=4, alpha=1e-4, beta=1e-4, gamma=1e-4, delta=1e-4, **kwargs):
        super().__init__(**kwargs)
        self._n_clusters = n_clusters
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def _update_W(self):
        """Please note that Eq. (9) in paper :cite:`2019:ren2` should be corrected to:

        .. math::

            \\begin{aligned}
            \\nabla_\\boldsymbol{W} = & \\boldsymbol{X}^{\\top} \\left(\\hat{\\boldsymbol{D}} - \\boldsymbol{D}\\right) + 2 \\lambda_1 \\boldsymbol{W} -
            \\boldsymbol{X}^{\\top} \\left(\\left(\\hat{\\boldsymbol{D}} - \\hat{\\boldsymbol{D}}^2\\right) \\odot
            \\boldsymbol{\\Gamma}_1\\right) \\boldsymbol{S}^{\\top} \\\\
            - & \\rho \\boldsymbol{X}^{\\top} \\left(\\left(\\hat{\\boldsymbol{D}} - \\hat{\\boldsymbol{D}}^2\\right) \\odot
            \\left(\\boldsymbol{D} - \\hat{\\boldsymbol{D}}\\boldsymbol{S} - \\boldsymbol{E}\\right)\\right) \\boldsymbol{S}^{\\top}\\text{,}
            \\end{aligned}

        where :math:`\\odot` denotes element-wise multiplication.
        """

        def _obj_func(w):
            self._W = w.reshape(self._n_features, self._n_outputs)
            return _update_W_numba(self._X, self._D, self._W, self._S, self._E,
                                   self._V, self._alpha, self._rho)

        optimize_result = minimize(_obj_func, self._W.reshape(-1, ), method='L-BFGS-B', jac=True)
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        self._update_S()
        self._update_E()

    def _update_S(self):

        def _obj_func(s):
            self._S = s.reshape(self._n_outputs, self._n_outputs)
            return _update_S_numba(self._X, self._D, self._W, self._S, self._E, self._Z,
                                   self._V, self._V2, self._P, self._sumP,
                                   self._n_clusters, self._delta, self._rho)

        optimize_result = minimize(_obj_func, self._S.reshape(-1, ), method='L-BFGS-B', jac=True)
        self._S = optimize_result.x.reshape(self._n_outputs, self._n_outputs)

    def _update_E(self):
        _, DSE = _get_D_pred_DSE(self._D, self._S, self._E, self._X, self._W)
        self._E = solvel21(DSE, self._beta / self._rho)

    def _update_Z(self):
        self._Z = svt(self._S - self._Z, self._gamma / self._rho)

    def _update_V(self):
        self._V, self._V2 = _update_V_numba(self._X, self._D, self._W, self._S, self._E,
                                            self._Z, self._V, self._V2, self._rho)

    @property
    def constraint(self):
        D_pred = _get_D_pred(self._X, self._W)
        return [[self._E, self._D - D_pred @ self._S],
                [self._Z, self._S]]

    @property
    def params(self):
        return [self._W, self._S, self._E, self._Z]

    @property
    def Vs(self):
        return [self._V, self._V2]

    def _get_default_model(self):
        _W = np.zeros((self._n_features, self._n_outputs))
        _Z = np.eye(self._n_outputs)
        _V = np.zeros((self._n_samples, self._n_outputs))
        return _W, _Z, _V

    def _before_train(self):
        c = KMeans(n_clusters=self._n_clusters).fit_predict(self._D)
        self._P = []
        self._sumP = 0.
        for i in range(self._n_clusters):
            temp = pairwise_euclidean(self._D[c == i].T)
            self._sumP += self._delta * np.sum(temp)
            self._P.append(temp)
        self._S = np.eye(self._n_outputs)
        self._E = np.zeros((self._n_samples, self._n_outputs))
        self._V2 = np.zeros((self._n_outputs, self._n_outputs))

    def fit(self, X, y,  rho=1e-4, **kwargs):
        return super().fit(X, y, rho=rho, **kwargs)

    def predict(self, X):
        return _get_D_pred(X, self._W)
