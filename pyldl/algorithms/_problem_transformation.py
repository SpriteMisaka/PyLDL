import numpy as np
from scipy.spatial.distance import pdist
from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import CalibratedClassifierCV

from pyldl.algorithms.base import BaseLDL


class _PT(BaseLDL):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = None

    def _preprocessing(self, X, y):
        m, c = y.shape[0], y.shape[1]
        Xr = np.repeat(X, c, axis=0)
        yr = np.tile(np.arange(c), m)
        p = y.reshape(-1) / m

        select = np.random.choice(m*c, size=m*c, p=p)

        return Xr[select], yr[select]

    def fit(self, X, y):
        super().fit(X, y)
        self._X, self._y = self._preprocessing(self._X, self._y)

        self._model.fit(self._X, self._y)

    def predict(self, X):
        return self._model.predict_proba(X)


class PT_Bayes(_PT):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = GaussianNB(var_smoothing=0.1)


class PT_SVM(_PT):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = CalibratedClassifierCV(LinearSVC())


class LDSVR(_PT):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = None

    def _preprocessing(self, X, y):
        y = -np.log(1. / np.clip(y, 1e-7, 1. - 1e-7) - 1.)
        return X, y

    def fit(self, X, y):
        BaseLDL.fit(self, X, y)
        self._model = MultiOutputRegressor(SVR(tol=1e-10, gamma=1./(2.*np.mean(pdist(self._X))**2)))

        self._X, self._y = self._preprocessing(self._X, self._y)
        self._model.fit(self._X, self._y)

    def predict(self, X):
        y_pred = 1. / (1. + np.exp(-self._model.predict(X)))
        return y_pred / np.sum(y_pred, axis=1).reshape(-1, 1)
