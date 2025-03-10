import numpy as np

from scipy.spatial.distance import pdist

from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import CalibratedClassifierCV

from pyldl.algorithms.base import BaseLDL


EPS = np.finfo(float).eps


class _PT(BaseLDL):
    """Base class for :class:`pyldl.algorithms.PT_Bayes` and :class:`pyldl.algorithms.PT_SVM`.

    PT refers to *problem transformation*.
    """

    def _preprocessing(self, X, D):
        m, c = D.shape[0], D.shape[1]
        Xr = np.repeat(X, c, axis=0)
        Lr = np.tile(np.arange(c), m)
        p = D.reshape(-1) / np.sum(D)

        select = np.random.choice(m*c, size=m*c, p=p)
        return Xr[select], Lr[select]

    def _get_default_model(self):
        pass

    def fit(self, X, D):
        super().fit(X, D)
        self._X, self._D = self._preprocessing(self._X, self._D)
        self._model = self._get_default_model()
        self._model.fit(self._X, self._D)
        return self

    def predict(self, X):
        return self._model.predict_proba(X)


class PT_Bayes(_PT):
    """:class:`PT-Bayes <pyldl.algorithms.PT_Bayes>` is proposed in paper :cite:`2016:geng`.
    """

    def _get_default_model(self):
        return GaussianNB(var_smoothing=.1)


class PT_SVM(_PT):
    """:class:`PT-SVM <pyldl.algorithms.PT_SVM>` is proposed in paper :cite:`2016:geng`.
    """

    def _get_default_model(self):
        return CalibratedClassifierCV(LinearSVC())


class LDSVR(_PT):
    """:class:`LDSVR <pyldl.algorithms.LDSVR>` is proposed in paper :cite:`2015:geng`.
    """

    def _preprocessing(self, X, D):
        D = -np.log(1. / np.clip(D, EPS, 1. - EPS) - 1.)
        return X, D

    def _get_default_model(self):
        return MultiOutputRegressor(SVR(tol=1e-7, gamma=1./(2.*np.mean(pdist(self._X))**2)))

    def predict(self, X):
        D_pred = 1. / (1. + np.exp(-self._model.predict(X)))
        return D_pred / np.sum(D_pred, axis=1).reshape(-1, 1)
