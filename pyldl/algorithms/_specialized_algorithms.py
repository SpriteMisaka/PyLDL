import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize, fsolve

from pyldl.algorithms.base import BaseLDL


EPS = np.finfo(np.float64).eps


class _SA(BaseLDL):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._W = None

    def _loss_function(self, y, y_pred):
        y_true = np.clip(y, EPS, 1)
        y_pred = np.clip(y_pred, EPS, 1)
        return -1 * np.sum(y_true * np.log(y_pred))

    def predict(self, X):
        return softmax(np.dot(X, self._W), axis=1)

    @property
    def W(self):
        if self._W is None:
            self._not_been_fit()
        return self._W


class SA_BFGS(_SA):

    def _obj_func(self, weights):
        W = weights.reshape(self._n_outputs, self._n_features).T
        y_pred = softmax(np.dot(self._X, W), axis=1)

        loss = self._loss_function(self._y, y_pred)
        grad = np.dot(self._X.T, y_pred - self._y).T.reshape(-1, )

        return loss, grad

    def fit(self, X, y, max_iterations=600, convergence_criterion=1e-6):
        super().fit(X, y)

        weights = np.random.uniform(-0.1, 0.1, self._n_features * self._n_outputs)

        optimize_result = minimize(self._obj_func, weights, method='L-BFGS-B', jac=True,
                                   options={'gtol': convergence_criterion,
                                            'disp': False, 'maxiter': max_iterations})

        self._W = optimize_result.x.reshape(self._n_outputs, self._n_features).T


class SA_IIS(_SA):

    def fit(self, X, y, max_iterations=600, convergence_criterion=1e-6):
        super().fit(X, y)

        weights = np.random.uniform(-0.1, 0.1, self._n_features * self._n_outputs)

        flag = True
        counter = 1
        W = weights.reshape(self._n_outputs, self._n_features).transpose()
        y_pred = softmax(np.dot(self._X, W), axis=1)

        while flag:
            delta = np.empty(shape=(self._X.shape[1], self._y.shape[1]), dtype=np.float32)
            for k in range(self._X.shape[1]):
                for j in range(self._y.shape[1]):
                    def func(x):
                        temp1 = np.sum(self._y[:, j] * self._X[:, k])
                        temp2 = np.sum(y_pred[:, j] * self._X[:, k] * \
                                       np.exp(x * np.sign(self._X[:, k]) * \
                                              np.sum(np.abs(self._X), axis=1)))
                        return temp1 - temp2
                    delta[k][j] = fsolve(func, .0, xtol=1e-10)

            l2 = self._loss_function(self._y, y_pred)
            weights += delta.transpose().ravel()
            y_pred = softmax(np.dot(self._X, W), axis=1)
            l1 = self._loss_function(self._y, y_pred)

            if l2 - l1 < convergence_criterion or counter >= max_iterations:
                flag = False

            W = weights.reshape(self._n_outputs, self._n_features).transpose()
            counter += 1

        self._W = W
