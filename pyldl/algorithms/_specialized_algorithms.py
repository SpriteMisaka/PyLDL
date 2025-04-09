import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize, fsolve

from pyldl.algorithms.base import BaseLDL
from pyldl.algorithms.utils import kl_divergence


class _SA(BaseLDL):
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

    def predict(self, X):
        return self._call(X)

    @property
    def W(self):
        if self._W is None:
            self._not_been_fit()
        return self._W


class SA_BFGS(_SA):
    """:class:`SA-BFGS <pyldl.algorithms.SA_BFGS>` is proposed in paper :cite:`2016:geng`.

    :term:`BFGS` is used as optimization algorithm.
    """

    def _obj_func(self, w):
        self._W = w.reshape(self._n_features, self._n_outputs)
        D_pred = self._call(self._X)

        loss = self._loss_function(self._D, D_pred)
        grad = (self._X.T @ (D_pred - self._D)).reshape(-1, )

        return loss, grad

    def fit(self, X, D, max_iterations=500, convergence_criterion=1e-7):
        super().fit(X, D)

        w0 = np.random.random(self._n_features * self._n_outputs)
        optimize_result = minimize(self._obj_func, w0, method='L-BFGS-B', jac=True,
                                   options={'gtol': convergence_criterion,
                                            'disp': False, 'maxiter': max_iterations})

        self._W = optimize_result.x.reshape(self._n_features, self._n_outputs)


class SA_IIS(_SA):
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

    def fit(self, X, D, max_iterations=500, convergence_criterion=1e-7):
        super().fit(X, D)

        self._W = np.random.random((self._n_features, self._n_outputs))

        flag = True
        counter = 1
        D_pred = self._call(self._X)

        XD = self._X.T @ self._D
        absX = np.sum(np.abs(self._X), axis=1)

        while flag:
            delta = np.empty(shape=(self._n_features, self._n_outputs), dtype=np.float32)
            for k in range(self._n_features):
                z = np.sign(self._X[:, k]) * absX
                for j in range(self._n_outputs):
                    def func(x):
                        return XD[k, j] - np.sum(D_pred[:, j] * self._X[:, k] * np.exp(x * z))
                    delta[k][j] = fsolve(func, .0)[0]

            l2 = self._loss_function(self._D, D_pred)
            self._W += delta
            D_pred = self._call(self._X)
            l1 = self._loss_function(self._D, D_pred)

            if l2 - l1 < convergence_criterion or counter >= max_iterations:
                flag = False
            counter += 1
