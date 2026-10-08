import copy
import heapq

import numpy as np

from scipy.special import expit, softmax
from sklearn.cluster import KMeans

from pyldl.algorithms.base import BaseEnsemble, BaseLDL

from pyldl.algorithms._tree import _Node, best_split
from pyldl.algorithms._rbm import train_rbm


EPS = np.finfo(np.float32).eps


class RG4LDL(BaseEnsemble):
    """:class:`RG4LDL <pyldl.algorithms.RG4LDL>` is proposed in paper :cite:`2025:tan`. 
    RG refers to *renormalization group*.

    This algorithm works well on the artificial dataset.
    """

    def __init__(self, estimator=None, *, n_hidden: int = 64, **kwargs):
        """
        :param estimator: The base estimator to be used for the final prediction. If None, :class:`SA-BFGS <pyldl.algorithms.SA_BFGS>` is used.
        :type estimator: :class:`BaseLDL <pyldl.algorithms.BaseLDL>`
        :param n_hidden: The number of hidden units in the restricted Boltzmann machine, default to 64.
        :type n_hidden: int
        """
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
    DF refers to *decomposition & fusion*.
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
        m, c = X.shape[0], self._n_outputs
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
    StructRF refers to *structured random forest*.
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

    class VectorTree(BaseLDL):
        """:class:`VectorTree` is proposed in paper :cite:`2016:xing`.
        """

        def __init__(self, n_leaves=2, min_to_split=6, **kwargs):
            super().__init__(**kwargs)
            self.n_leaves = n_leaves
            self.min_to_split = min_to_split

        def fit(self, X, D, F=None):
            super().fit(X, D)
            self._P = softmax(np.zeros_like(self._D) if F is None else F, axis=1)
            self._G = self._P - self._D
            self._order = np.argsort(self._X, axis=0, kind='stable').T
            self._tree, heap, n_leaves = [], [], 1
            self._node(np.arange(self._n_samples), heap)
            while n_leaves < self.n_leaves and heap:
                _, k, feature, value, left, right = heapq.heappop(heap)
                self._tree[k][:4] = feature, value, self._node(left, heap), self._node(right, heap)
                n_leaves += 1
            for node in self._tree:
                if node[0] < 0:
                    node[4] = self._leaf(node[4])
            del self._P, self._G, self._order
            return self

        def predict(self, X):
            return softmax(self._predict_F(X), axis=1)

        def _predict_F(self, X):
            F = np.zeros((X.shape[0], self._n_outputs))
            stack = [(0, np.arange(X.shape[0]))]
            while stack:
                k, rows = stack.pop()
                feature, value, left, right, a = self._tree[k]
                if feature < 0:
                    F[rows] = a
                else:
                    mask = X[rows, feature] < value
                    stack += [(left, rows[mask]), (right, rows[~mask])]
            return F

        def _pair(self, idx):
            g, P = np.sum(self._G[idx], axis=0), self._P[idx]
            r = np.argmax(-g)
            h = np.sum(P * (1 - P), axis=0)
            h = h[r] + h + 2 * np.sum(P[:, [r]] * P, axis=0)
            gain = (g[r] - g) ** 2 / np.where(h == 0, 1, h)
            gain[r] = -1
            return r, np.argmax(gain)

        def _uv(self, idx):
            r, s = self._pair(idx)
            P = self._P
            u = self._G[:, s] - self._G[:, r]
            v = P[:, r] * (1 - P[:, r]) + P[:, s] * (1 - P[:, s]) + 2 * P[:, r] * P[:, s]
            return r, s, u, v

        @staticmethod
        def _gain(u, v):
            return u ** 2 / (2 * np.where(v == 0, 1, v))

        def _node(self, idx, heap):
            k = len(self._tree)
            self._tree.append([-1, 0., 0, 0, idx])
            if len(idx) < self.min_to_split:
                return k
            _, _, u, v = self._uv(idx)
            mask = np.zeros(self._n_samples, dtype=bool)
            mask[idx] = True
            S = self._order[mask[self._order]].reshape(self._order.shape[0], -1)
            x = np.take_along_axis(self._X.T, S, axis=1)
            U, V = np.cumsum(u[S], axis=1), np.cumsum(v[S], axis=1)
            UL, VL = U[:, :-1], V[:, :-1]
            gain = self._gain(UL, VL) + self._gain(U[:, -1:] - UL, V[:, -1:] - VL)
            gain = np.where(x[:, :-1] < x[:, 1:], gain, -np.inf)
            if not np.isfinite(gain).any():
                return k
            feature, i = np.unravel_index(np.argmax(gain), gain.shape)
            expected = gain[feature, i] - self._gain(U[0, -1], V[0, -1])
            heapq.heappush(heap, (-expected, k, feature, (x[feature, i] + x[feature, i + 1]) / 2, S[feature, :i + 1], S[feature, i + 1:]))
            return k

        def _leaf(self, idx):
            r, s, u, v = self._uv(idx)
            gamma = np.sum(u[idx]) / (np.sum(v[idx]) or 1.)
            a = np.zeros(self._n_outputs)
            a[r], a[s] = gamma, -gamma
            return a

    def __init__(self, estimator=None, n_estimators=100, mode=None, **kwargs):
        from sklearn.tree import DecisionTreeRegressor
        if estimator is None:
            estimator = self.VectorTree(n_leaves=20) if mode == 'AOSO' else DecisionTreeRegressor()
        super().__init__(estimator, n_estimators, **kwargs)
        self._mode = mode

    def _calculate_Fj(self, f):
        return self._learning_rate * ((self._n_outputs - 1) / self._n_outputs) * (f - np.mean(f, axis=1, keepdims=True))

    def fit(self, X, D, learning_rate=0.05):
        super().fit(X, D)
        self._estimators = []
        self._learning_rate = learning_rate
        self._F = np.zeros((self._n_samples, self._n_outputs), dtype=np.float32)
        for i in range(self._n_estimators):
            if self._mode == 'AOSO':
                model = copy.deepcopy(self._estimator).fit(self._X, self._D, self._F)
                self._F += self._learning_rate * model._predict_F(self._X)
                self._estimators.append(model)
                continue
            P = softmax(self._F, axis=1)
            H = P * (1 - P)
            Z = (self._D - P) / H
            f = np.zeros((self._n_samples, self._n_outputs))
            self._estimators.append([])
            for j in range(self._n_outputs):
                model = copy.deepcopy(self._estimator)
                model.fit(self._X, Z[:, j], sample_weight=H[:, j])
                f[:, j] = model.predict(self._X)
                self._estimators[i].append(model)
            self._F += self._calculate_Fj(f)
        return self

    def predict(self, X):
        F = np.zeros((X.shape[0], self._n_outputs), dtype=np.float32)   
        for i in range(self._n_estimators):
            if self._mode == 'AOSO':
                F += self._learning_rate * self._estimators[i]._predict_F(X)
                continue
            f = np.stack([self._estimators[i][j].predict(X) for j in range(self._n_outputs)], axis=1)
            F += self._calculate_Fj(f)
        return softmax(F, axis=1)
