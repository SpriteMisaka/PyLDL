import numpy as np

from scipy.spatial.distance import pdist

from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import CalibratedClassifierCV

from pyldl.algorithms.base import BaseLDL


class _PT(BaseLDL):
    """Base class for :class:`pyldl.algorithms.PT_Bayes` and :class:`pyldl.algorithms.PT_SVM`.

    PT refers to *problem transformation*.
    """

    def _preprocessing(self, X, y):
        m, c = y.shape[0], y.shape[1]
        Xr = np.repeat(X, c, axis=0)
        yr = np.tile(np.arange(c), m)
        p = y.reshape(-1) / np.sum(y)

        select = np.random.choice(m*c, size=m*c, p=p)
        return Xr[select], yr[select]

    def _get_default_model(self):
        pass

    def fit(self, X, y):
        super().fit(X, y)
        self._X, self._y = self._preprocessing(self._X, self._y)
        self._model = self._get_default_model()
        self._model.fit(self._X, self._y)
        return self

    def predict(self, X):
        return self._model.predict_proba(X)


class PT_Bayes(_PT):
    """:class:`PT-Bayes <pyldl.algorithms.PT_Bayes>` is proposed in paper :cite:`2016:geng`.
    """

    def _get_default_model(self):
        return GaussianNB(var_smoothing=0.1)


class PT_SVM(_PT):
    """:class:`PT-SVM <pyldl.algorithms.PT_SVM>` is proposed in paper :cite:`2016:geng`.
    """

    def _get_default_model(self):
        return CalibratedClassifierCV(LinearSVC())


class LDSVR(_PT):
    """:class:`LDSVR <pyldl.algorithms.LDSVR>` is proposed in paper :cite:`2015:geng`.
    """

    def _preprocessing(self, X, y):
        y = -np.log(1. / np.clip(y, 1e-7, 1. - 1e-7) - 1.)
        return X, y

    def _get_default_model(self):
        return MultiOutputRegressor(SVR(tol=1e-10, gamma=1./(2.*np.mean(pdist(self._X))**2)))

    def predict(self, X):
        y_pred = 1. / (1. + np.exp(-self._model.predict(X)))
        return y_pred / np.sum(y_pred, axis=1).reshape(-1, 1)
