import copy

import numpy as np

from pyldl.algorithms.base import BaseEnsemble
from pyldl.algorithms.utils import sort_loss


class DF_LDL(BaseEnsemble):
    """:class:`DF-LDL <pyldl.algorithms.DF_LDL>` is proposed in paper :cite:`2021:gonzalez`.
    """

    def __init__(self, estimator=None, random_state=None):
        from ._specialized_algorithms import SA_BFGS
        if estimator is None:
            estimator = SA_BFGS()
        super().__init__(estimator, None, random_state)

    def fit(self, X, y):
        super().fit(X, y)

        L = {}

        for i in range(self._n_outputs):
            for j in range(i + 1, self._n_outputs):

                ss1 = []
                ss2 = []

                for k in range(self._n_samples):
                    if self._D[k, i] >= self._D[k, j]:
                        ss1.append(k)
                    else:
                        ss2.append(k)

                l1 = copy.deepcopy(self._estimator)
                l1.fit(self._X[ss1], self._D[ss1])
                L[f"{str(i)},{str(j)}"] = copy.deepcopy(l1)

                l2 = copy.deepcopy(self._estimator)
                l2.fit(self._X[ss2], self._D[ss2])
                L[f"{str(j)},{str(i)}"] = copy.deepcopy(l2)

        self._estimators = L

        from ._algorithm_adaptation import AA_KNN
        self._knn = AA_KNN()
        self._knn.fit(self._X, self._D)

    def predict(self, X):

        m, c = X.shape[0], self._D.shape[1]
        p_knn = self._knn.predict(X)
        p = np.zeros((m, c), dtype=np.float32)

        for k in range(m):
            for i in range(c):
                for j in range(i + 1, c):

                    if p_knn[k, i] >= p_knn[k, j]:
                        l = self._estimators[f"{str(i)},{str(j)}"]
                    else:
                        l = self._estimators[f"{str(j)},{str(i)}"]

                    p[k] += l.predict(X[k].reshape(1, -1)).reshape(-1)

        return p / (c * (c - 1) / 2)


class AdaBoostLDL(BaseEnsemble):

    def __init__(self, estimator=None, n_estimators=10, alpha=1., **kwargs):
        from ._specialized_algorithms import SA_BFGS
        if estimator is None:
            estimator = SA_BFGS()
        super().__init__(estimator, n_estimators, **kwargs)
        self._alpha = alpha

    def fit(self, X, D, loss=sort_loss):
        super().fit(X, D)
        p = np.ones((self._n_samples,)) / self._n_samples
        self._loss = np.zeros((self._n_estimators, self._n_samples))
        self._estimators = []
        for i in range(self._n_estimators):
            select = np.random.choice(self._n_samples, size=self._n_samples, p=p)
            X_train, D_train = self._X[select], self._D[select]

            model = copy.deepcopy(self._estimator)
            model.fit(X_train, D_train)
            self._estimators.append(copy.deepcopy(model))

            D_pred = model.predict(self._X)
            self._loss[i] = loss(D, D_pred, reduction=None)
            p += self._alpha * (self._loss[i] / np.sum(self._loss))
            p /= np.sum(p)

    def predict(self, X):
        w = np.sum(self._loss, axis=1)
        w /= np.sum(w)
        D = np.zeros((X.shape[0], self._n_outputs))
        for i in range(self._n_estimators):
            D += w[i] * self._estimators[i].predict(X)
        return D
