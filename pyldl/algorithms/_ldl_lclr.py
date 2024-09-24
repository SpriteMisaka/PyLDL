import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from sklearn.cluster import KMeans

from pyldl.algorithms.base import BaseADMM, BaseLDL
from pyldl.algorithms.utils import kl_divergence, svt, solvel21, pairwise_euclidean


class LDL_LCLR(BaseADMM, BaseLDL):
    """:class:`LDL-LCLR <pyldl.algorithms.LDL_LCLR>` is proposed in paper :cite:`2019:ren2`.

    :term:`ADMM` is used as the optimization algorithm.
    """

    def _update_W(self):
        """Please note that Eq. (9) in paper :cite:`2019:ren2` should be corrected to:

        .. math::

            \\begin{aligned}
            \\nabla_\\boldsymbol{W} = & \\boldsymbol{X}^{\\text{T}} \\left(\\hat{\\boldsymbol{D}} - \\boldsymbol{D}\\right) + 2 \\lambda_1 \\boldsymbol{W} -
            \\boldsymbol{X}^{\\text{T}} \\left(\\left(\\hat{\\boldsymbol{D}} - \\hat{\\boldsymbol{D}}^2\\right) \\odot
            \\boldsymbol{\\Gamma}_1\\right) \\boldsymbol{S}^{\\text{T}} \\\\
            - & \\rho \\boldsymbol{X}^{\\text{T}} \\left(\\left(\\hat{\\boldsymbol{D}} - \\hat{\\boldsymbol{D}}^2\\right) \\odot
            \\left(\\boldsymbol{D} - \\hat{\\boldsymbol{D}}\\boldsymbol{S} - \\boldsymbol{E}\\right)\\right) \\boldsymbol{S}^{\\text{T}}\\text{,}
            \\end{aligned}

        where :math:`\\odot` denotes element-wise multiplication.
        """

        def _obj_func(w):
            self._W = w.reshape(self._n_features, self._n_outputs)
            XW = self._X @ self._W
            y_pred = softmax(XW, axis=1)
            ySE = self._y - y_pred @ self._S - self._E
            yy2 = y_pred - y_pred ** 2

            def _W_loss():
                kl = kl_divergence(self._y, y_pred, reduction=np.sum)
                inn = np.sum(self._V * (ySE))
                fro1 = np.linalg.norm(self._W, 'fro') ** 2
                fro2 = self._rho * np.linalg.norm(ySE, 'fro') ** 2 / 2.
                return kl + inn + self._alpha * fro1 + fro2

            def _W_grad():
                grad = self._X.T @ (y_pred - self._y)
                grad += 2 * self._alpha * self._W
                grad -= self._X.T @ (yy2 * self._V) @ self._S.T
                grad -= self._rho * self._X.T @ (yy2 * ySE) @ self._S.T
                return grad.reshape(-1, )

            return _W_loss(), _W_grad()

        w0 = self._W.reshape(-1, ).copy()
        optimize_result = minimize(_obj_func, w0, method='L-BFGS-B', jac=True)
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        self._update_S()
        self._update_E()

    def _update_S(self):

        def _obj_func(s):
            self._S = s.reshape(self._n_outputs, self._n_outputs)
            XW = self._X @ self._W
            y_pred = softmax(XW, axis=1)
            ySE = self._y - y_pred @ self._S - self._E

            def _S_loss():
                inn1 = np.sum(self._V * (ySE))
                inn2 = np.sum(self._V2 * (self._S - self._Z))
                inn = inn1 + inn2
                fro1 = np.linalg.norm(self._S - self._Z, 'fro') ** 2
                fro2 = np.linalg.norm(ySE, 'fro') ** 2
                fro = self._rho * (fro1 + fro2) / 2.
                pairwise = 0.
                for i in range(self._n_clusters):
                    pairwise -= np.sum(self._S * self._P[i])
                return inn + fro + self._delta * pairwise

            def _S_grad():
                grad = - self._V.T @ y_pred + self._V2.T
                grad += self._rho * (self._S - self._Z - y_pred.T @ ySE)
                grad -= self._sumP
                return grad.reshape(-1, )

            return _S_loss(), _S_grad()

        s0 = self._S.reshape(-1, ).copy()
        optimize_result = minimize(_obj_func, s0, method='L-BFGS-B', jac=True)
        self._S = optimize_result.x.reshape(self._n_outputs, self._n_outputs)

    def _update_E(self):
        XW = self._X @ self._W
        y_pred = softmax(XW, axis=1)
        ySE = self._y - y_pred @ self._S - self._E
        self._E = solvel21(ySE, self._beta / self._rho)

    def _update_Z(self):
        self._Z = svt(self._S - self._Z, self._gamma / self._rho)

    def _update_V(self):
        XW = self._X @ self._W
        y_pred = softmax(XW, axis=1)
        ySE = self._y - y_pred @ self._S - self._E
        self._V = self._V + self._rho * ySE
        self._V2 = self._V2 + self._rho * (self._S - self._Z)

    @property
    def constraint(self):
        XW = self._X @ self._W
        y_pred = softmax(XW, axis=1)
        return [[self._E, self._y - y_pred @ self._S],
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
        _V = np.zeros((self._X.shape[0], self._n_outputs))
        return _W, _Z, _V

    def _before_train(self):
        c = KMeans(n_clusters=self._n_clusters).fit_predict(self._y)
        self._P = []
        self._sumP = 0.
        for i in range(self._n_clusters):
            temp = pairwise_euclidean(self._y[c == i].T)
            self._sumP += self._delta * np.sum(temp)
            self._P.append(temp)
        self._S = np.eye(self._n_outputs)
        self._E = np.zeros((self._X.shape[0], self._n_outputs))
        self._V2 = np.zeros((self._n_outputs, self._n_outputs))

    def fit(self, X, y, n_clusters=4,
            alpha=1e-4, beta=1e-4, gamma=1e-4, delta=1e-4, rho=1e-4, **kwargs):
        self._n_clusters = n_clusters
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        return super().fit(X, y, rho=rho, **kwargs)

    def predict(self, X):
        return softmax(X @ self._W, axis=1)
