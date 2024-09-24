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

        m, c = self._y.shape[0], self._y.shape[1]
        L = {}

        for i in range(c):
            for j in range(i + 1, c):

                ss1 = []
                ss2 = []

                for k in range(m):
                    if self._y[k, i] >= self._y[k, j]:
                        ss1.append(k)
                    else:
                        ss2.append(k)

                l1 = copy.deepcopy(self._estimator)
                l1.fit(self._X[ss1], self._y[ss1])
                L[str(i)+","+str(j)] = copy.deepcopy(l1)

                l2 = copy.deepcopy(self._estimator)
                l2.fit(self._X[ss2], self._y[ss2])
                L[str(j)+","+str(i)] = copy.deepcopy(l2)

        self._estimators = L

        from ._algorithm_adaptation import AA_KNN
        self._knn = AA_KNN()
        self._knn.fit(self._X, self._y)

    def predict(self, X):

        m, c = X.shape[0], self._y.shape[1]
        p_knn = self._knn.predict(X)
        p = np.zeros((m, c), dtype=np.float32)

        for k in range(m):
            for i in range(c):
                for j in range(i + 1, c):

                    if p_knn[k, i] >= p_knn[k, j]:
                        l = self._estimators[str(i)+","+str(j)]
                    else:
                        l = self._estimators[str(j)+","+str(i)]

                    p[k] += l.predict(X[k].reshape(1, -1)).reshape(-1)

        return p / (c * (c - 1) / 2)


class AdaBoostLDL(BaseEnsemble):

    def __init__(self, estimator=None, n_estimators=10, random_state=None):
        from ._specialized_algorithms import SA_BFGS
        if estimator is None:
            estimator = SA_BFGS()
        super().__init__(estimator, n_estimators, random_state)

    def fit(self, X, y, loss=sort_loss, alpha=1.):
        super().fit(X, y)

        m = self._X.shape[0]
        p = np.ones((m,)) / m

        self._loss = np.zeros((self._n_estimators, m))
        self._estimators = []
        for i in range(self._n_estimators):
            select = np.random.choice(m, size=m, p=p)
            X_train, y_train = self._X[select], self._y[select]

            model = copy.deepcopy(self._estimator)
            model.fit(X_train, y_train)
            self._estimators.append(copy.deepcopy(model))

            y_pred = model.predict(self._X)
            self._loss[i] = loss(y, y_pred, reduction=None)
            p += alpha * (self._loss[i] / np.sum(self._loss))
            p /= np.sum(p)

    def predict(self, X):
        w = np.sum(self._loss, axis=1)
        w /= np.sum(w)
        y = np.zeros((X.shape[0], self._n_outputs))
        for i in range(self._n_estimators):
            y += w[i] * self._estimators[i].predict(X)
        return y
