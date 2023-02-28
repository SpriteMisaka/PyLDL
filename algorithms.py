import os

import numpy as np
import numba

from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import tensorflow as tf
from tensorflow import keras

from metrics import score


class _BaseLDL():

    def __init__(self, random_state=None):
        if random_state != None:
            np.random.seed(random_state)

        self._n_features = None
        self._n_outputs = None

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._n_features = self._X.shape[1]
        self._n_outputs = self._y.shape[1]

    def predict(self, _):
        pass

    def score(self, X, y,
              metrics=["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]):
        return score(y, self.predict(X), metrics=metrics)

    def _not_been_fit(self):
        raise ValueError("The model has not yet been fit. "
                         "Try to call 'fit()' first with some training data.")
    
    @property
    def n_features(self):
        if self._n_features == None:
            self._not_been_fit()
        return self._n_features

    @property
    def n_outputs(self):
        if self._n_outputs == None:
            self._not_been_fit()
        return self._n_outputs


class BaseLDL(_BaseLDL, BaseEstimator):
    
    def __init__(self, random_state=None):
        super().__init__(random_state)


class BaseDeepLDL(_BaseLDL, keras.Model):
    
    def __init__(self, random_state=None):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        super(_BaseLDL, self).__init__(random_state)
        super(keras.Model, self).__init__()
        if random_state != None:
            tf.random.set_seed(random_state)


class _SA(BaseLDL):

    def __init__(self,
                 maxiter=600,
                 convergence_criterion=1e-6,
                 random_state=None):

        super().__init__(random_state)

        self.maxiter = maxiter
        self.convergence_criterion = convergence_criterion

        self._W = None
        
    def _object_fun(self, weights):
        W = weights.reshape(self._n_outputs, self._n_features).transpose()
        y_pred = softmax(np.dot(self._X, W), axis=1)
        
        func_loss = self._loss(y_pred)
        func_grad = self._gradient(y_pred)
        
        return func_loss, func_grad

    def _gradient(self, y_pred):
        grad = np.dot(self._X.T, y_pred - self._y)
        return grad.transpose().reshape(-1, )

    def _loss(self, y_pred):
        y_true = np.clip(self._y, 1e-7, 1)
        y_pred = np.clip(y_pred, 1e-7, 1)
        return -1 * np.sum(y_true * np.log(y_pred))

    def _specialized_alg(self, _):
        pass

    def fit(self, X, y):
        super().fit(X, y)

        weights = np.random.uniform(-0.1, 0.1, self._n_features * self._n_outputs)
        self._W = self._specialized_alg(weights)

    def predict(self, X):
        return softmax(np.dot(X, self._W), axis=1)

    @property
    def W(self):
        if self._W == None:
            self._not_been_fit()
        return self._W


class SA_BFGS(_SA):
    
    def _specialized_alg(self, weights):
        optimize_result = minimize(self._object_fun, weights, method='L-BFGS-B', jac=True,
                                   options={'gtol': self.convergence_criterion,
                                            'disp': False, 'maxiter': self.maxiter})
        return optimize_result.x.reshape(self._n_outputs, self._n_features).transpose()


@numba.jit(nopython=True, fastmath=True)
def _solve(_X, _y, y_pred):

    delta = np.empty(shape=(_X.shape[1], _y.shape[1]), dtype=np.float32)
    for k in range(_X.shape[1]):
        for j in range(_y.shape[1]):
            
            temp1 = np.sum(_y[:, j] * _X[:, k])
            temp2 = np.sum(y_pred[:, j] * _X[:, k] * \
                           np.exp(np.sign(_X[:, k]) * np.sum(np.abs(_X), axis=1)))
            delta[k][j] = np.log(temp1 / (temp2 + 1e-8))

    return delta


class SA_IIS(_SA):

    def _specialized_alg(self, weights):

        flag = True
        counter = 1
        W = weights.reshape(self._n_outputs, self._n_features).transpose()
        y_pred = softmax(np.dot(self._X, W), axis=1)

        while flag:
            delta = _solve(self._X, self._y, y_pred)

            l2 = self._loss(y_pred)
            weights += delta.transpose().ravel()
            y_pred = softmax(np.dot(self._X, W), axis=1)
            l1 = self._loss(y_pred)

            if l2 - l1 < self.convergence_criterion or counter >= self.maxiter:
                flag = False
            
            W = weights.reshape(self._n_outputs, self._n_features).transpose()
            counter += 1

        return W


class AA_KNN(BaseLDL):

    def __init__(self,
                 k=5,
                 random_state=None):

        super().__init__(random_state)

        self.k = k
        self._model = NearestNeighbors(n_neighbors=self.k)
    
    def fit(self, X, y):
        super().fit(X, y)
        self._model.fit(self._X)
        
    def predict(self, X):
        _, inds = self._model.kneighbors(X)
        return np.average(self._y[inds], axis=1)


class AA_BP(BaseDeepLDL):

    def __init__(self, random_state=None):
        super().__init__(random_state)

    def fit(self, X, y, latent=None, epochs=600, **fit_params):
        super().fit(X, y)

        if latent == None:
            latent = self._n_features * 3 // 2

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=(self._n_features,)),
                                        keras.layers.Dense(latent, activation='sigmoid'),
                                        keras.layers.Dense(self._n_outputs, activation='softmax')])
        self._model.compile(loss="mean_squared_error")

        self._model.fit(self._X, self._y, verbose=0, epochs=epochs, **fit_params)

    def predict(self, X):
        return self._model(X)


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
        self._model = BernoulliNB()
        

class PT_SVM(_PT):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = CalibratedClassifierCV(LinearSVC())


__all__ = ["SA_BFGS", "SA_IIS", "AA_KNN", "AA_BP", "PT_Bayes", "PT_SVM"]


if __name__ == '__main__':

    from sklearn.model_selection import KFold
    import pandas as pd
    from utils import load_dataset
    from tqdm import *

    seed = 114514
    X, y = load_dataset('SJAFFE')

    columns = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]
    
    for method in __all__:
        df = pd.DataFrame(columns=columns)
        print(method)
        for i in tqdm(range(10)):

            kfold = KFold(n_splits=10, shuffle=True, random_state=seed+i)
            for train_index, test_index in tqdm(kfold.split(X), leave=False):

                model = eval(f'{method}()')
                model.fit(X[train_index], y[train_index])

                df.loc[len(df.index)] = model.score(X[test_index], y[test_index], metrics=columns)
                df.to_excel(f'{method}.xlsx', index=False)
