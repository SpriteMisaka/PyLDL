import copy

import numpy as np

from scipy.special import expit, softmax
from sklearn.cluster import KMeans

from pyldl.algorithms.base import BaseEnsemble, BaseLDL
from pyldl.algorithms.utils import sort_loss

from pyldl.algorithms._tree import _Node, best_split
from pyldl.algorithms._rbm import train_rbm


EPS = np.finfo(np.float32).eps


class RG4LDL(BaseEnsemble):
    """:class:`RG4LDL <pyldl.algorithms.RG4LDL>` is proposed in paper :cite:`2025:tan`.
    """

    def __init__(self, estimator=None, *, n_hidden: int = 64, **kwargs):
        from ._specialized_algorithms import SA_BFGS
        if estimator is None:
            estimator = SA_BFGS()
        super().__init__(estimator, None, **kwargs)
        self.n_hidden = n_hidden

    def fit(self, X, D, *, rbm_iterations=10, rbm_batch_size=64, rbm_learning_rate=1e-2,
            rbm_init_temperature=1000, rbm_final_temperature=10, **kwargs):
        super().fit(X, D, **kwargs)
        H, self._W, self._b, self._c = train_rbm(X,
            np.random.normal(size=(self._n_features, self.n_hidden)),
            np.random.normal(size=(self._n_features,)),
            np.random.normal(size=(self.n_hidden,)),
            rbm_iterations, rbm_batch_size, rbm_learning_rate,
            rbm_init_temperature, rbm_final_temperature
        )
        self._estimators = [copy.deepcopy(self._estimator)]
        self._estimators[0].fit(H, self._D)

    def predict(self, X):
        H = expit(X @ self._W + self._c)
        return self._estimators[0].predict(H)


class DF_LDL(BaseEnsemble):
    """:class:`DF-LDL <pyldl.algorithms.DF_LDL>` is proposed in paper :cite:`2021:gonzalez`.
    """

    def __init__(self, estimator=None, *, k: int = 5, **kwargs):
        from ._specialized_algorithms import SA_BFGS
        if estimator is None:
            estimator = SA_BFGS()
        super().__init__(estimator, None, **kwargs)
        self.k = k

    def fit(self, X, D):
        super().fit(X, D)
        self._estimators = [[None] * self._n_outputs for _ in range(self._n_outputs)]
        for i in range(self._n_outputs):
            for j in range(i + 1, self._n_outputs):
                ss1 = [k for k in range(self._n_samples) if self._D[k, i] >= self._D[k, j]]
                ss2 = [k for k in range(self._n_samples) if self._D[k, i] < self._D[k, j]]

                est1 = copy.deepcopy(self._estimator)
                est1.fit(self._X[ss1], self._D[ss1])
                self._estimators[i][j] = est1

                est2 = copy.deepcopy(self._estimator)
                est2.fit(self._X[ss2], self._D[ss2])
                self._estimators[j][i] = est2

        from ._algorithm_adaptation import AA_KNN
        self._knn = AA_KNN(k=self.k)
        self._knn.fit(self._X, self._D)

    def predict(self, X):
        m, c = X.shape[0], self._D.shape[1]
        p_knn = self._knn.predict(X)
        p = np.zeros((m, c), dtype=np.float32)
        for k in range(m):
            for i in range(c):
                for j in range(i + 1, c):
                    est = self._estimators[i][j] if p_knn[k, i] >= p_knn[k, j] else self._estimators[j][i]
                    p[k] += est.predict(X[k].reshape(1, -1)).reshape(-1)
        return p / (c * (c - 1) / 2)


class StructRF(BaseEnsemble):
    """:class:`StructRF` is proposed in paper :cite:`2018:chen`.
    """

    class StructTree(BaseLDL):
        """:class:`StructTree` is proposed in paper :cite:`2018:chen`.
        """

        def __init__(self, max_depth=20, min_to_split=5, alpha=.25, beta=8., **kwargs):
            super().__init__(**kwargs)
            self.max_depth = max_depth
            self.min_to_split = min_to_split
            self.alpha = alpha
            self.beta = beta

        def fit(self, X, D):
            super().fit(X, D)
            self._C = np.zeros(self._n_samples, dtype=np.int32)
            self._root = self._leaf(np.arange(len(self._X), dtype=np.int32))
            self._split_recursively(self._root, 0)

        def predict(self, X):
            results = np.zeros((X.shape[0], self._n_outputs), dtype=np.float32)
            for i in range(len(X)):
                node = self._root
                while not node.is_leaf:
                    node = node.left if X[i][node.feature] <= node.value else node.right
                results[i] = node.prediction
            return results

        def _split_recursively(self, node: _Node, depth: int):
            if not self._can_split(node, depth):
                node.prediction = np.mean(self._D[node.indices], axis=0)
                return
            self._C[node.indices] = KMeans(n_clusters=2).fit_predict(self._D[node.indices])
            feature, value, left, right = best_split(self._X, self._C, node.indices, self.alpha, self.beta)
            node.split(feature, value, self._leaf(left), self._leaf(right))
            self._split_recursively(node.left, depth + 1)
            self._split_recursively(node.right, depth + 1)

        def _can_split(self, node: _Node, depth: int):
            return (
                (self.max_depth is None or depth < self.max_depth) and
                len(node.indices) >= self.min_to_split
            )

        def _leaf(self, indices):
            return _Node(indices)

    def __init__(self, estimator=None, n_estimators=20, sampling_ratio=.8, **kwargs):
        if estimator is None:
            estimator = self.StructTree()
        super().__init__(estimator, n_estimators, **kwargs)
        self.sampling_ratio = sampling_ratio

    def fit(self, X, D):
        super().fit(X, D)
        self._estimators = []
        for _ in range(self._n_estimators):
            select = np.random.choice(self._n_samples, size=int(self._n_samples * self.sampling_ratio), replace=False)
            model = copy.deepcopy(self._estimator)
            model.fit(X[select], D[select])
            self._estimators.append(copy.deepcopy(model))

    def predict(self, X):
        results = np.zeros((X.shape[0], self._n_outputs), dtype=np.float32)
        for model in self._estimators:
            results += model.predict(X) / self._n_estimators
        return results


class LDLogitBoost(BaseEnsemble):
    """:class:`LDLogitBoost` is proposed in paper :cite:`2016:xing`.
    """

    def __init__(self, estimator=None, n_estimators=100, **kwargs):
        from sklearn.tree import DecisionTreeRegressor
        if estimator is None:
            estimator = DecisionTreeRegressor()
        super().__init__(estimator, n_estimators, **kwargs)

    def _calculate_Fj(self, f):
        return self._learning_rate * ((self._n_outputs - 1) / self._n_outputs) * (f - np.mean(f))

    def fit(self, X, D, learning_rate=0.05):
        super().fit(X, D)
        self._estimators = []
        self._learning_rate = learning_rate
        self._F = np.zeros((self._n_samples, self._n_outputs), dtype=np.float32)
        for i in range(self._n_estimators):
            P = softmax(self._F, axis=1)
            H = P * (1 - P)
            Z = (self._D - P) / H
            for j in range(self._n_outputs):
                model = copy.deepcopy(self._estimator)
                model.fit(self._X, Z[:, j], sample_weight=H[:, j])
                f = model.predict(self._X)
                self._estimators.append([])
                self._estimators[i].append(copy.deepcopy(model))
                self._F[:, j] += self._calculate_Fj(f)
        return self

    def predict(self, X):
        F = np.zeros((X.shape[0], self._n_outputs), dtype=np.float32)   
        for i in range(self._n_estimators):
            for j in range(self._n_outputs):
                f = self._estimators[i][j].predict(X)
                F[:, j] += self._calculate_Fj(f)
        return softmax(F, axis=1)


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
