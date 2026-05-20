from typing import Protocol

import numpy as np

from pyldl.algorithms.base import Base, BaseLDL, BaseGLD


EPS = np.finfo(float).eps


class _PT(Base):
    """Base class for :class:`pyldl.algorithms.PT_Bayes` and :class:`pyldl.algorithms.PT_SVM`.

    PT refers to *problem transformation*.
    """

    class _PTModel(Protocol):
        def fit(self, X: np.ndarray, y: np.ndarray) -> '_PT._PTModel': ...
        def predict(self, X: np.ndarray) -> np.ndarray: ...
        def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def _preprocessing(self, X: np.ndarray, D: np.ndarray):
        n, c = D.shape[0], D.shape[1]
        Xr = np.repeat(X, c, axis=0)
        Yr = np.tile(np.arange(c), n)
        p = D.reshape(-1) / np.sum(D)
        select = np.random.choice(n*c, size=n*c, p=p)
        return Xr[select], Yr[select]

    def _get_default_model(self) -> _PTModel:
        raise NotImplementedError(
            "The '_get_default_model()' method is not implemented."
        )

    def fit(self, X, target):
        super().fit(X, target)
        self._Xr, self._Yr = self._preprocessing(self._X, self._target)
        self._model = self._get_default_model()
        self._model.fit(self._Xr, self._Yr)
        return self

    def predict(self, X):
        return self._model.predict_proba(X)


class PT_Bayes(_PT, BaseLDL):
    """:class:`PT-Bayes <pyldl.algorithms.PT_Bayes>` is proposed in paper :cite:`2016:geng`.
    """

    def __init__(self, var_smoothing: float = .1, **kwargs):
        super().__init__(**kwargs)
        self.var_smoothing = var_smoothing

    def _get_default_model(self):
        from sklearn.naive_bayes import GaussianNB
        priors = np.bincount(self._Yr) / len(self._Yr)
        return GaussianNB(priors=priors, var_smoothing=self.var_smoothing)


class PT_SVM(_PT, BaseLDL):
    """:class:`PT-SVM <pyldl.algorithms.PT_SVM>` is proposed in paper :cite:`2016:geng`.
    """

    def _get_default_model(self):
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(LinearSVC())


class _Reg2LDL(_PT, BaseLDL):

    def _preprocessing(self, X: np.ndarray, D: np.ndarray):
        D = -np.log((1. + EPS) / np.clip(D, EPS, 1.) - 1.)
        return X, D

    def predict(self, X: np.ndarray):
        from pyldl.algorithms.utils import normalize
        D_pred = 1. / (1. + np.exp(-self._model.predict(X)))
        return normalize(D_pred)


class _Reg2GLD(_PT, BaseGLD):

    def _preprocessing(self, X: np.ndarray, G: np.ndarray):
        G = np.arctanh(np.clip(G, -1. + 1e-7, 1. - 1e-7))
        return X, G

    def predict(self, X: np.ndarray):
        return np.tanh(self._model.predict(X))


class _SVRMixin(Base):
    def _get_default_model(self):
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        from scipy.spatial.distance import pdist
        gamma = 1. / (2. * np.mean(pdist(self._X)) ** 2)
        return MultiOutputRegressor(SVR(tol=1e-7, gamma=gamma))


class LDSVR(_SVRMixin, _Reg2LDL):
    """:class:`LDSVR <pyldl.algorithms.LDSVR>` is proposed in paper :cite:`2015:geng`.
    """


class GLD_SVR(_SVRMixin, _Reg2GLD):
    pass
