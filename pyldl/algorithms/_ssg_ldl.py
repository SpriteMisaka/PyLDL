import numpy as np
from sklearn.neighbors import NearestNeighbors

from pyldl.algorithms.base import BaseLDL


class SSG_LDL(BaseLDL):
    """:class:`SSG-LDL <pyldl.algorithms.SSG_LDL>` is proposed in paper :cite:`2021:gonzalez2`.
    """

    def __init__(self, n=300, k=5, fx=0.5, fy=0.5, random_state=None):
        super().__init__(random_state)

        self._n = n
        self._k = k
        self._fx = fx
        self._fy = fy

    def _select_sample(self):
        total_dist = np.sum(self._dist)
        r = np.random.rand() * total_dist
        for i in range(self._n_samples):
            r -= self._dist[i]
            if r < 0:
                return i

    def _create_synthetic_sample(self, i):
        _, index = self._knn.kneighbors(np.concatenate([self._X[i], self._D[i]]).reshape(1, -1))
        nnarray = index.reshape(-1)
        nn = np.random.randint(1, self._k)

        dif = self._X[nnarray[nn-1]] - self._X[i]
        gap = np.random.random(self._X[0].shape)

        X = self._X[i] + gap * dif
        D = np.average(self._D[nnarray], axis=0)

        self._new_X = np.concatenate([self._new_X, X.reshape(1, -1)])
        self._new_D = np.concatenate([self._new_D, D.reshape(1, -1)])

    def fit_transform(self, X, D, synthetic_only=False):
        super().fit(X, D)

        repeated_X = np.repeat([self._X], self._n_samples, axis=0).transpose(1, 0, 2)
        dist_X = np.sum(np.linalg.norm(repeated_X - self._X, axis=2), axis=1) / self._n_samples
        repeated_D = np.repeat([self._D], self._n_samples, axis=0).transpose(1, 0, 2)
        dist_D = np.sum(np.linalg.norm(repeated_D - self._D, axis=2), axis=1) / self._n_samples
        self._dist = self._fx * dist_X + self._fy * dist_D

        self._knn = NearestNeighbors(n_neighbors=self._k)
        self._knn.fit(np.concatenate([self._X, self._D], axis=1))

        self._new_X = self._X.copy()
        self._new_D = self._D.copy()

        t = self._n_samples * self._n // 100

        for _ in range(t):
            self._create_synthetic_sample(self._select_sample())

        start = 0 if synthetic_only else self._n_samples
        return self._new_X[start:], self._new_D[start:]
