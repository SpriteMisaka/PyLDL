import copy
import types
from functools import reduce

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.spatial.distance import cosine, pdist

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseEnsemble, BaseAdam, BaseDeepLDL

from pyldl.algorithms._ldl_lrr import LDL_LRR
from pyldl.algorithms._ldl_scl import LDL_SCL
from pyldl.algorithms.loss_function_engineering import qfd2, cjs


class _S_LDL:

    @staticmethod
    def nested_len(combi):
        return reduce(lambda count, l: count + len(l), combi, 0)

    class SubtaskConstruction:
        def __init__(self, t=10, lam=0.2):
            self._t = t
            self._lam = lam

        def _loss(self, W):
            W = expit(W).reshape((self._t, len(self._D_bar)))
            alpha = np.exp(-np.mean(W * self._D_bar, axis=1))
            beta = pdist(W, lambda u, v: np.abs(1 - cosine(u, v)))
            return np.mean(alpha) + self._lam * np.mean(beta)

        def fit(self, D):
            self._D_bar = np.average(D, axis=0)
            W_init = np.random.normal(size=(self._t, D.shape[1]))
            result = minimize(self._loss, W_init.flatten(), method='L-BFGS-B', jac=None)
            self._W = expit(result.x.reshape((self._t, D.shape[1])))
            return self

        def transform(self):
            W_bin = (self._W > 0.9).astype(int)
            masks = {
                tuple(W_bin[i]) for i in range(W_bin.shape[0])
                if not np.all(W_bin[i] == 1) and np.sum(W_bin[i] == 1) != 1
            }
            indices_list = []
            for row in list(masks):
                indices = np.where(np.array(row) == 1)[0]
                indices_list.append(indices)
            return indices_list

    def __init__(self, combi, t, lam):
        self._combi = combi
        self._t = t
        self._lam = lam

    def _generate_subtasks(self, D):
        self._combi = self.SubtaskConstruction(self._t, self._lam).fit(D).transform()


class Shallow_S_LDL(_S_LDL, BaseEnsemble):

    def __init__(self, estimator=None, combi=None, t=10, lam=0.2, **kwargs):
        from ._problem_transformation import LDSVR
        if estimator is None:
            estimator = LDSVR()
        n_estimators = (t + 1) if combi is None else (len(combi) + 1)
        _S_LDL.__init__(self, combi, t, lam)
        BaseEnsemble.__init__(self, estimator, n_estimators, **kwargs)

    def fit(self, X, D):
        super().fit(X, D)
        if self._combi is None:
            self._generate_subtasks(self._D)
        f = copy.deepcopy(self._estimator)
        Z = np.copy(self._X)
        if self._combi:
            fs = [copy.deepcopy(self._estimator) for _ in range(len(self._combi))]
            for i in range(len(self._combi)):
                fs[i].fit(self._X, self._D[:, self._combi[i]])
                Z = np.concatenate((Z, fs[i].predict(self._X)), axis=1)
        f.fit(Z, self._D)
        self._estimator = ([f] + fs) if self._combi else [f]

    def predict(self, X):
        Z = np.copy(X)
        if self._combi:
            for i in range(1, len(self._combi) + 1):
                Z = np.concatenate((Z, self._estimator[i].predict(X)), axis=1)
        return self._estimator[0].predict(Z)


class _DeepSLDL(_S_LDL, BaseAdam, BaseDeepLDL):

    @staticmethod
    @tf.function
    def loss_function(D, D_pred):
        return tf.math.reduce_mean(keras.losses.kl_divergence(D, D_pred))

    def __init__(self, *, combi=None, t=10, lam=0.2, n_hidden=64,
                 mu=.1, spec_function=None, **kwargs):
        _S_LDL.__init__(self, combi, t, lam)
        n_latent = self.nested_len(combi) if combi else None
        BaseDeepLDL.__init__(self, n_hidden, n_latent, **kwargs)
        self._mu = mu
        self._spec_function = spec_function

    def _call(self, X):
        rep = self._model["encoder"](X)
        latent = self._model["decoder_S"](rep)
        features = tf.concat([rep, latent], axis=1)
        outputs = self._model["decoder"](features)
        return rep, latent, features, outputs

    def _get_default_model(self):
        encoder = self.get_2layer_model(self._n_features, self._n_hidden, activation=keras.layers.LeakyReLU())
        decoder_S = self.get_2layer_model(self._n_hidden, self._n_latent, activation=keras.layers.LeakyReLU())
        decoder = self.get_3layer_model(self._n_hidden + self._n_latent, self._n_hidden, self._n_outputs,
                                        hidden_activation=keras.layers.LeakyReLU(), output_activation=None)
        return {"encoder": encoder, "decoder_S": decoder_S, "decoder": decoder}

    def _before_train(self):
        if self._combi is None:
            self._generate_subtasks(self._D)
        if self._n_latent is None:
            self._n_latent = self.nested_len(self._combi) if self._combi else 0

        if self._spec_function:
            class_name = f"LDL_{self._spec_function}"
            if cls := globals().get(class_name):
                cls._before_train(self)
            elif isinstance(self._spec_function, types.FunctionType):
                self.loss_function = self._spec_function
            else:
                raise ValueError(f"Unsupported spec_function: {self._spec_function}")

        temp = [tf.clip_by_value(tf.gather(self._D, axis=1, indices=c), 1e-7, 1.0) for c in self._combi]
        self._real_task = [i / (tf.reshape(tf.reduce_sum(i, axis=1), (-1, 1))) for i in temp]
        self._weights = [tf.math.reduce_sum(tf.gather(self._D, axis=1, indices=c), axis=1) for c in self._combi]

    def _spec_pred(self, outputs, start, end):
        return keras.activations.softmax(outputs)

    def _spec_loss(self, D_pred, start, end):
        return 0.

    @tf.function
    def _loss(self, X, D, start, end):
        _, latent, _, outputs = self._call(X)
        le, ls, sub_start = 0., 0., 0

        D_pred = self._spec_pred(outputs, start, end)
        le = self._spec_loss(D_pred, start, end)

        for i in range(len(self._combi)):
            temp = keras.activations.softmax(latent[:, sub_start:sub_start+len(self._combi[i])])
            mae = keras.losses.mean_absolute_error(self._real_task[i][start:end], temp)
            ls += tf.math.reduce_mean(self._weights[i][start:end] * mae)
            sub_start += len(self._combi[i])
        l = self.loss_function(D, D_pred)
        return l + le + self._mu * ls

    def predict(self, X):
        _, _, _, outputs = self._call(X)
        return keras.activations.softmax(outputs)


class S_LRR(_DeepSLDL):

    def __init__(self, alpha=1e-3, **kwargs):
        super().__init__(spec_function='LRR', **kwargs)
        self._alpha = alpha

    def _spec_loss(self, D_pred, start, end):
        return self._alpha * LDL_LRR.ranking_loss(D_pred, self._P[start:end], self._W[start:end])


class S_SCL(_DeepSLDL):

    def __init__(self, n_clusters=5, alpha=1e-3, beta=1e-6, **kwargs):
        super().__init__(spec_function='SCL', **kwargs)
        self._n_clusters = n_clusters
        self._alpha = alpha
        self._beta = beta

    def _spec_pred(self, outputs, start, end):
        return keras.activations.softmax(outputs + self._C[start:end] @ self._W)

    def _spec_loss(self, D_pred, start, end):
        corr, barr = LDL_SCL.scl_loss(D_pred, self._P, self._C[start:end])
        return self._alpha * corr + self._beta * barr

    def predict(self, X):
        _, _, _, outputs = self._call(X)
        C = LDL_SCL.construct_C(X, self._X, self._C)
        return keras.activations.softmax(outputs + C @ self._W)


class S_KLD(_DeepSLDL):

    def __init__(self, **kwargs):
        super().__init__(spec_function=None, **kwargs)


class S_QFD2(_DeepSLDL):

    def __init__(self, **kwargs):
        super().__init__(spec_function=qfd2, **kwargs)


class S_CJS(_DeepSLDL):

    def __init__(self, **kwargs):
        super().__init__(spec_function=cjs, **kwargs)
