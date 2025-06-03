import copy

import numpy as np

from numba import jit

from scipy.special import expit
from sklearn.cluster import KMeans

from pyldl.algorithms.base import BaseEnsemble, BaseLDL
from pyldl.algorithms.utils import sort_loss

from pyldl.algorithms._tree import _Node, best_split


EPS = np.finfo(np.float32).eps


@jit(nopython=True)
def _sigmoid_numba(Z):
    return 1 / (1 + np.exp(-Z))


@jit(nopython=True)
def _average_diff_numba(Z0, Z1):
    dZ = np.zeros_like(Z0[0])
    for i in range(Z0.shape[0]):
        dZ += (Z0[i] - Z1[i])
    dZ /= Z0.shape[0]
    return dZ


@jit(nopython=True)
def _energy_numba(X, H, W, b, c):
    return -(np.sum(X @ b.reshape(-1, 1)) + np.sum(H @ c.reshape(-1, 1)) + np.sum((X @ W) * H))


@jit(nopython=True)
def _train_rbm_numba(X, W, b, c, iterations, batch_size, lr, init_t, final_t):
    for iter in range(iterations):
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            X0 = X[start:end]
            H0 = _sigmoid_numba(X0 @ W + c)
            H0_sample = (H0 > np.random.rand(*(H0.shape))).astype(np.float64)
            X1 = _sigmoid_numba(H0_sample @ W.T + b)
            H1 = _sigmoid_numba(X1 @ W + c)
            H1_sample = (H1 > np.random.rand(*(H1.shape))).astype(np.float64)

            dE = _energy_numba(X0, H0_sample, W, b, c) - _energy_numba(X1, H1_sample, W, b, c)
            t = max(final_t, init_t * .7 ** iter)
            if dE < 0 or np.random.rand() < np.exp(-dE / t):
                X0 = X1

            W += lr * ((X0.T @ H0_sample - X1.T @ H1_sample) / X0.shape[0])
            b += lr * _average_diff_numba(X0, X1)
            c += lr * _average_diff_numba(H0_sample, H1_sample)
    H = _sigmoid_numba(X @ W + c)
    return H, W, b, c


class RG4LDL(BaseEnsemble):
    """:class:`RG4LDL <pyldl.algorithms.RG4LDL>` is proposed in paper :cite:`2025:tan`.
    """

    def __init__(self, estimator=None, *, n_hidden: int = 64, **kwargs):
        from ._specialized_algorithms import SA_BFGS
        if estimator is None:
            estimator = SA_BFGS()
        super().__init__(estimator, None, **kwargs)
        self._n_hidden = n_hidden

    def fit(self, X, D, *, rbm_iterations=10, rbm_batch_size=64, rbm_learning_rate=1e-2,
            rbm_init_temperature=1000, rbm_final_temperature=10, **kwargs):
        super().fit(X, D, **kwargs)
        H, self._W, self._b, self._c = _train_rbm_numba(X,
            np.random.normal(size=(self._n_features, self._n_hidden)),
            np.random.normal(size=(self._n_features,)),
            np.random.normal(size=(self._n_hidden,)),
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
        self._knn = AA_KNN(self.k)
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


class StructRF(BaseEnsemble):
    """:class:`StructRF` is proposed in paper :cite:`2018:chen`.
    """

    class StructTree(BaseLDL):
        """:class:`StructTree` is proposed in paper :cite:`2018:chen`.
        """

        def __init__(self, max_depth=20, min_to_split=5, alpha=.25, beta=8., **kwargs):
            super().__init__(**kwargs)
            self._max_depth = max_depth
            self._min_to_split = min_to_split
            self._alpha = alpha
            self._beta = beta

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
            feature, value, left, right = best_split(self._X, self._C, node.indices, self._alpha, self._beta)
            node.split(feature, value, self._leaf(left), self._leaf(right))
            self._split_recursively(node.left, depth + 1)
            self._split_recursively(node.right, depth + 1)

        def _can_split(self, node: _Node, depth: int):
            return (
                (self._max_depth is None or depth < self._max_depth) and
                len(node.indices) >= self._min_to_split
            )

        def _leaf(self, indices):
            return _Node(indices)

    def __init__(self, estimator=None, n_estimators=20, sampling_ratio=.8,
                 max_depth=20, min_to_split=5, alpha=.25, beta=8., **kwargs):
        if estimator is None:
            estimator = self.StructTree(max_depth=max_depth, min_to_split=min_to_split, alpha=alpha, beta=beta)
        super().__init__(estimator, n_estimators, **kwargs)
        self._sampling_ratio = sampling_ratio

    def fit(self, X, D):
        super().fit(X, D)
        self._estimators = []
        for _ in range(self._n_estimators):
            select = np.random.choice(self._n_samples, size=int(self._n_samples * self._sampling_ratio), replace=False)
            model = copy.deepcopy(self._estimator)
            model.fit(X[select], D[select])
            self._estimators.append(copy.deepcopy(model))

    def predict(self, X):
        results = np.zeros((X.shape[0], self._n_outputs), dtype=np.float32)
        for model in self._estimators:
            results += model.predict(X) / self._n_estimators
        return results


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
