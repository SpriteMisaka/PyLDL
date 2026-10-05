import numpy as np

from numba import jit

from scipy.optimize import minimize
from scipy.special import softmax
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

from pyldl.algorithms.base import BaseADMM, BaseLDL
from pyldl.algorithms.utils import svt, solvel21


@jit(nopython=True)
def _get_log_D_pred(X, W):
    XW = X @ W
    for i in range(XW.shape[0]):
        XW[i] -= np.max(XW[i])
    return XW - np.log(np.sum(np.exp(XW), axis=1)).reshape(-1, 1)


@jit(nopython=True)
def _get_D_pred(X, W):
    return np.exp(_get_log_D_pred(X, W))


@jit(nopython=True)
def _get_D_pred_DSE(D, S, E, X, W):
    D_pred = _get_D_pred(X, W)
    return D_pred, D - D_pred @ S - E


@jit(nopython=True)
def _update_W_numba(X, D, W, S, E, V, alpha, rho):
    log_D_pred = _get_log_D_pred(X, W)
    D_pred, DSE = _get_D_pred_DSE(D, S, E, X, W)
    G = -(V + rho * DSE) @ S.T
    D_flat = D.reshape(-1, )
    mask = D_flat > 0
    kl = np.sum(D_flat[mask] * (np.log(D_flat[mask]) - log_D_pred.reshape(-1, )[mask]))
    inn = np.sum(V * DSE)
    fro1 = np.linalg.norm(W) ** 2
    fro2 = rho * np.linalg.norm(DSE) ** 2 / 2.
    loss = kl + inn + alpha * fro1 + fro2
    grad = X.T @ (D_pred - D)
    grad += 2 * alpha * W
    grad += X.T @ (D_pred * (G - np.sum(D_pred * G, axis=1).reshape(-1, 1)))
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
    grad = - D_pred.T @ V + V2
    grad += rho * (S - Z - D_pred.T @ DSE)
    grad -= sumP
    return loss, grad.reshape(-1, )


@jit(nopython=True)
def _update_V_numba(X, D, W, S, E, Z, V, V2, rho):
    _, DSE = _get_D_pred_DSE(D, S, E, X, W)
    return V + rho * DSE, V2 + rho * (S - Z)


class LDL_LCLR(BaseADMM, BaseLDL):
    """:class:`LDL-LCLR <pyldl.algorithms.LDL_LCLR>` is proposed in paper :cite:`2019:ren2`. 
    LC and LR refer to *label correlation* and *low-rank*, respectively.

    :term:`ADMM` is used as the optimization algorithm.
    """

    def __init__(self, n_clusters=4, alpha=1e-4, beta=1e-4, gamma=1e-4, delta=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _update_W(self):
        r"""The gradient of Eq. (7) in paper :cite:`2019:ren2`, correcting Eq. (9), is:

        .. math::

            \begin{aligned}
            \hat{\boldsymbol{D}} &= \operatorname{softmax}(\boldsymbol{X}\boldsymbol{W}), \\
            \boldsymbol{R} &= \boldsymbol{D} - \hat{\boldsymbol{D}}\boldsymbol{S} - \boldsymbol{E}, \\
            \boldsymbol{G} &= -(\boldsymbol{\Gamma}_1 + \rho\boldsymbol{R})\boldsymbol{S}^{\top}, \\
            \nabla_{\boldsymbol{W}} &= \boldsymbol{X}^{\top}[
            \hat{\boldsymbol{D}} - \boldsymbol{D} + \hat{\boldsymbol{D}}\odot(
            \boldsymbol{G} - ((\hat{\boldsymbol{D}}\odot\boldsymbol{G})\boldsymbol{1})
            \boldsymbol{1}^{\top})] + 2\lambda_1\boldsymbol{W}.
            \end{aligned}

        Here :math:`\odot` denotes element-wise multiplication.
        """

        def _obj_func(w):
            self._W = w.reshape(self._n_features, self._n_outputs)
            return _update_W_numba(self._X, self._D, self._W, self._S, self._E,
                                   self._V, self.alpha, self._rho)

        optimize_result = minimize(_obj_func, self._W.reshape(-1, ), method='L-BFGS-B', jac=True)
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        self._update_S()
        self._update_E()

    def _update_S(self):
        r"""The gradient of Eq. (8) in paper :cite:`2019:ren2`, correcting Eq. (10), is:

        .. math::

            \begin{aligned}
            \boldsymbol{R} &= \boldsymbol{D} - \hat{\boldsymbol{D}}\boldsymbol{S} - \boldsymbol{E}, \\
            \nabla_{\boldsymbol{S}} &= -\hat{\boldsymbol{D}}^{\top}\boldsymbol{\Gamma}_1
            + \boldsymbol{\Gamma}_2
            + \rho(\boldsymbol{S} - \boldsymbol{Z}
            - \hat{\boldsymbol{D}}^{\top}\boldsymbol{R})
            - \frac{\lambda_4}{2}\sum_v\boldsymbol{P}_v.
            \end{aligned}

        Here :math:`(\boldsymbol{P}_v)_{mn} = \|\boldsymbol{D}^{v}_{\cdot m} - \boldsymbol{D}^{v}_{\cdot n}\|_2^2`.
        """

        def _obj_func(s):
            self._S = s.reshape(self._n_outputs, self._n_outputs)
            return _update_S_numba(self._X, self._D, self._W, self._S, self._E, self._Z,
                                   self._V, self._V2, self._P, self._sumP,
                                   self.n_clusters, self.delta, self._rho)

        optimize_result = minimize(_obj_func, self._S.reshape(-1, ), method='L-BFGS-B', jac=True)
        self._S = optimize_result.x.reshape(self._n_outputs, self._n_outputs)

    def _update_E(self):
        D_pred = _get_D_pred(self._X, self._W)
        self._E = solvel21(self._D - D_pred @ self._S + self._V / self._rho, self.beta / self._rho)

    def _update_Z(self):
        self._Z = svt(self._S + self._V2 / self._rho, self.gamma / self._rho)

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
        c = KMeans(n_clusters=self.n_clusters).fit_predict(self._D)
        self._P = []
        self._sumP = np.zeros((self._n_outputs, self._n_outputs))
        for i in range(self.n_clusters):
            D_cluster = self._D[c == i].T
            temp = cdist(D_cluster, D_cluster, metric='sqeuclidean')
            self._sumP += self.delta * temp
            self._P.append(temp)
        self._S = np.eye(self._n_outputs)
        self._E = np.zeros((self._n_samples, self._n_outputs))
        self._V2 = np.zeros((self._n_outputs, self._n_outputs))

    def fit(self, X, y, rho=1e-4, **kwargs):
        return super().fit(X, y, rho=rho, **kwargs)

    def predict(self, X):
        XW = X @ self._W
        return softmax(XW, axis=1)
