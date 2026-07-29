import numpy as np

from pyldl.algorithms.base.shallow import Base, BaseLDL, BaseGLD


EPS = np.finfo(np.float32).eps


class _AA_KNN(Base):
    def __init__(self, *, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def fit(self, X: np.ndarray, target: np.ndarray):
        from sklearn.neighbors import NearestNeighbors
        super().fit(X, target)
        self._knn = NearestNeighbors(n_neighbors=self.k).fit(self._X)
        return self

    def predict(self, X):
        _, inds = self._knn.kneighbors(X)
        return np.average(self._target[inds], axis=1)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class AA_KNN(_AA_KNN, BaseLDL):
    """:class:`AA-kNN <pyldl.algorithms.AA_KNN>` is proposed in paper :cite:`2016:geng`.
    """
    pass


class GLD_KNN(_AA_KNN, BaseGLD):
    pass


class BD_LDL(BaseLDL):
    """:class:`BD-LDL <pyldl.algorithms.BD_LDL>` is proposed in paper :cite:`2021:liu2`.
    """

    def __init__(self, *, alpha: float = 1e-3, beta: float = 1e-2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, D):
        from scipy.linalg import solve_sylvester
        super().fit(X, D)
        A = self._X.T @ self._X + self.beta * np.eye(self._n_features)
        B = self.alpha * self._D.T @ self._D
        C = (1 + self.alpha) * self._X.T @ self._D
        self._W = solve_sylvester(A, B, C)
        return self

    def predict(self, X):
        from pyldl.algorithms.utils import proj
        return proj(X @ self._W)
