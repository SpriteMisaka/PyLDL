import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize, fsolve

from sklearn.covariance import LedoitWolf

from pyldl.algorithms.base import Base, BaseIter, BaseLDL, BaseGLD
from pyldl.algorithms.utils import kl_divergence


EPS = np.finfo(np.float32).eps


class _SA(Base):
    """Base class for :class:`pyldl.algorithms.SA_IIS` and :class:`pyldl.algorithms.SA_BFGS`.

    SA refers to *specialized algorithms*, where :term:`MaxEnt` is employed as model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._W = None

    def _loss_function(self, D, D_pred):
        return kl_divergence(D, D_pred, reduction=np.sum)

    def _call(self, X):
        return softmax(X @ self._W, axis=1)

    def get_weights(self):
        return self._W.copy()

    def set_weights(self, weights):
        self._W = weights.copy()

    def fit(self, X, target, max_iterations=500, convergence_criterion=1e-7):
        super().fit(X, target)
        self._max_iterations = max_iterations
        self._convergence_criterion = convergence_criterion
        self._W = np.random.random((self._n_features, self._n_outputs))

    def predict(self, X):
        return self._call(X)

    @property
    def W(self):
        if self._W is None:
            self._not_been_fit()
        return self._W


class SA_BFGS(_SA, BaseLDL):
    """:class:`SA-BFGS <pyldl.algorithms.SA_BFGS>` is proposed in paper :cite:`2016:geng`.

    :term:`BFGS` is used as optimization algorithm.
    """

    def _obj_func(self, w):
        self._W = w.reshape(self._n_features, self._n_outputs)
        D_pred = self._call(self._X)

        loss = self._loss_function(self._D, D_pred)
        grad = (self._X.T @ (D_pred - self._D)).reshape(-1, )

        return loss, grad

    def fit(self, X, D, **kwargs):
        super().fit(X, D, **kwargs)
        optimize_result = minimize(self._obj_func, self._W.reshape(-1,), method='L-BFGS-B', jac=True,
                                   options={'gtol': self._convergence_criterion,
                                            'maxiter': self._max_iterations})
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        return self


class SA_IIS(BaseIter, _SA, BaseLDL):
    """:class:`SA-IIS <pyldl.algorithms.SA_IIS>` is proposed in paper :cite:`2016:geng`.

    :term:`IIS` is used as optimization algorithm.

    IIS-LLD is the early version of :class:`SA-IIS <pyldl.algorithms.SA_IIS>`. See also:

    .. bibliography:: bib/ldl/references.bib bib/app/cv/references.bib
        :filter: False
        :labelprefix: SA-IIS-
        :keyprefix: sa-iis-

        2015:zhang
        2013:geng
        2010:geng
    """

    def fit(self, X, D, max_iterations=500, *args, **kwargs):
        return super().fit(
            X, D, max_iterations=max_iterations, *args, **kwargs
        )

    def _before_train(self, **kwargs):
        self._D_pred = self._call(self._X)
        self._XD = self._X.T @ self._D
        self._absX = np.sum(np.abs(self._X), axis=1)

    def _run_iteration(self):
        delta = np.empty(shape=(self._n_features, self._n_outputs), dtype=np.float32)
        for k in range(self._n_features):
            z = np.sign(self._X[:, k]) * self._absX
            for j in range(self._n_outputs):
                def func(x):
                    return self._XD[k, j] - np.sum(
                        self._D_pred[:, j] * self._X[:, k] * np.exp(x * z)
                    )
                delta[k][j] = fsolve(func, .0)[0]

        previous_loss = self._loss_function(self._D, self._D_pred)
        self._W += delta
        self._D_pred = self._call(self._X)
        loss = self._loss_function(self._D, self._D_pred)
        return loss, previous_loss - loss < self._convergence_criterion


class GLD_BFGS(_SA, BaseGLD):

    def _loss_function(self, G, G_pred):
        diff = G - G_pred
        left = diff @ self._sigma_inv
        mahalanobis_squared = np.sum(left * diff, axis=1)
        return np.mean(mahalanobis_squared)

    def _call(self, X):
        return np.tanh(X @ self._W)

    def _obj_func(self, w):
        self._W = w.reshape(self._n_features, self._n_outputs)
        G_pred = self._call(self._X)

        loss = self._loss_function(self._target, G_pred)
        grad = (-2 / self._n_samples) * (
            self._X.T @ ((self._target - G_pred) * (1 - G_pred ** 2)) @ self._sigma_inv
        ).reshape(-1, )

        self._current_iteration += 1
        tmp = float(self._current_iteration) / self._max_iterations
        if tmp >= .5 and self._current_iteration % 100 == 0:
            self._alpha = (tmp - .5) * 2
            eps = self._target - G_pred
            cov = LedoitWolf().fit(eps).covariance_
            self._sigma_inv = (
                self._alpha * np.linalg.inv(cov) +
                (1 - self._alpha) * np.eye(self._n_outputs)
            )

        return loss, grad

    def fit(self, X, G, **kwargs):
        super().fit(X, G, **kwargs)
        self._alpha = 0.
        self._current_iteration = 0
        self._sigma_inv = np.eye(self._n_outputs)

        limit = np.sqrt(6. / (self._n_features + self._n_outputs))
        self._W = np.random.uniform(-limit, limit, (self._n_features, self._n_outputs))

        optimize_result = minimize(
            self._obj_func,
            self._W.reshape(-1,),
            method='L-BFGS-B',
            jac=True,
            options={
                'gtol': self._convergence_criterion,
                'maxiter': self._max_iterations,
            }
        )
        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)
        return self


class LALOT(BaseIter, _SA, BaseLDL):
    """:class:`LALOT <pyldl.algorithms.LALOT>` is proposed in paper :cite:`2018:zhao`.
    """

    def __init__(self, *, alpha: float = .2, beta: float = 200., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def _compute_metric(self, K):
        diag = np.diag(K)
        M = diag[:, None] + diag[None, :] - 2 * K
        M = np.abs(M)
        M = M / np.max(M)
        return M

    def _call(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return softmax(X1 @ self._W, axis=1)

    def fit(self, X, D, max_iterations=500, *args, **kwargs):
        return super().fit(
            X, D, max_iterations=max_iterations, *args, **kwargs
        )

    def _before_train(self, sinkhorn_iterations=200, learning_rate=1e-4, **kwargs):
        self._sinkhorn_iterations = sinkhorn_iterations
        self._learning_rate = learning_rate
        self._W = np.random.random((self._n_features + 1, self._n_outputs))
        self._K0 = self._D.T @ self._D
        self._K1 = self._K0 / np.max(self._K0)
        self._D_pred = self._call(self._X)

    def _run_iteration(self):
        M = self._compute_metric(self._K1)
        K = np.exp(-self.alpha * M - 1)
        P = np.zeros((self._n_outputs, self._n_outputs), dtype=np.float32)
        G = np.zeros((self._n_outputs, self._n_features + 1), dtype=np.float32)

        for i in range(self._n_samples):
            x = np.concatenate((self._X[i], [1]), axis=0)
            d = self._D[i]
            d_pred = self._D_pred[i]
            u = np.ones(self._n_outputs, dtype=np.float32)
            for _ in range(self._sinkhorn_iterations):
                u = d_pred / (K @ (d / (K.T @ u)))
            v = d / (K.T @ u)
            P += np.diag(u) @ K @ np.diag(v)

            Glh = np.log(u) / self.alpha - np.log(u).sum() / (self.alpha * self._n_outputs)
            Gi = ((np.eye(self._n_outputs) - d_pred) @ Glh)[:, None] * d_pred[:, None] * x[None, :]
            G += Gi

        previous_loss = self._loss_function(self._D, self._D_pred)
        self._W -= self._learning_rate * G.T
        self._D_pred = self._call(self._X)
        loss = self._loss_function(self._D, self._D_pred)

        Gfk = -2 * P - np.diag(np.diag(-2 * P)) + np.diag(np.sum(P, axis=1) + np.sum(P, axis=0) - 2 * np.diag(P))
        self._K1 = self._K0 - (1 / self.beta) * Gfk
        Sigma, UK = np.linalg.eig((self._K1 + self._K1.T) / 2)
        self._K1 = UK @ np.diag(np.maximum(Sigma, 0)) @ UK.T
        return loss, previous_loss - loss < self._convergence_criterion
