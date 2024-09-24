import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDLClassifier, BaseGD, BaseBFGS


EPS = np.finfo(np.float32).eps


class LDL4C(BaseBFGS, BaseDeepLDLClassifier):
    """:class:`LDL4C <pyldl.algorithms.LDL4C>` is proposed in paper :cite:`2019:wang3`.
    """

    @tf.function
    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        y_pred = keras.activations.softmax(self._X @ theta)
        top2 = tf.gather(y_pred, self._top2, axis=1, batch_dims=1)
        margin = tf.reduce_sum(tf.maximum(0., 1. - (top2[:, 0] - top2[:, 1]) / self._rho))
        mae = keras.losses.mean_absolute_error(self._y, y_pred)
        return tf.reduce_sum(self._entropy * mae) + self._alpha * margin + self._beta * self._l2_reg(theta)

    def _before_train(self):
        self._top2 = tf.math.top_k(self._y, k=2)[1]
        self._entropy = tf.cast(-tf.reduce_sum(self._y * tf.math.log(self._y) + EPS, axis=1), dtype=tf.float32)

    def fit(self, X, y, alpha=1e-2, beta=0., rho=1e-2, **kwargs):
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        return super().fit(X, y, **kwargs)


class LDL_HR(BaseBFGS, BaseDeepLDLClassifier):
    """:class:`LDL-HR <pyldl.algorithms.LDL_HR>` is proposed in paper :cite:`2021:wang3`.
    """

    @tf.function
    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        y_pred = keras.activations.softmax(self._X @ theta)

        highest = tf.gather(y_pred, self._highest, axis=1, batch_dims=1)
        rest = tf.gather(y_pred, self._rest, axis=1, batch_dims=1)
        margin = tf.reduce_sum(tf.maximum(0., 1. - (highest - rest) / self._rho))

        real_rest = tf.gather(self._y, self._rest, axis=1, batch_dims=1)
        rest_mae = tf.reduce_sum(keras.losses.mean_absolute_error(real_rest, rest))

        mae = tf.reduce_sum(keras.losses.mean_absolute_error(self._l, y_pred))

        return mae + self._alpha * margin + self._beta * rest_mae + self._gamma * self._l2_reg(theta)

    def _before_train(self):
        temp = tf.math.top_k(self._y, k=self._n_outputs)[1]
        self._highest = temp[:, 0:1]
        self._rest = temp[:, 1:]
        self._l = tf.one_hot(tf.reshape(self._highest, -1), self._n_outputs)

    def fit(self, X, y, alpha=1e-2, beta=1e-2, gamma=0., rho=1e-2, **kwargs):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._rho = rho
        return super().fit(X, y, **kwargs) 


class LDLM(BaseGD, BaseDeepLDLClassifier):
    """:class:`LDLM <pyldl.algorithms.LDLM>` is proposed in paper :cite:`2021:wang`.
    """

    @tf.function
    def _loss(self, X, y, start, end):
        y_pred = self._model(X)

        pred_margin = tf.reduce_sum(tf.clip_by_value(
            tf.reduce_sum(tf.abs(self._l[start:end] - y_pred), axis=1) - self._rho,
            0., float('inf')))

        highest = tf.gather(y_pred, self._highest[start:end], axis=1, batch_dims=1)
        rest = tf.gather(y_pred, self._rest[start:end], axis=1, batch_dims=1)
        label_margin = tf.reduce_sum(tf.maximum(0., 1. - (highest - rest) / self._rho))

        second_margin = tf.reduce_sum(tf.clip_by_value(
            tf.reduce_sum(tf.abs((y - y_pred) * self._neg_l[start:end]), axis=1) - self._second_margin[start:end],
            0., float('inf')))

        return pred_margin + self._alpha * label_margin + \
            self._beta * second_margin + self._gamma * self._l2_reg(self._model)

    def _before_train(self):
        temp = tf.math.top_k(self._y, k=self._n_outputs)[1]
        self._highest = temp[:, 0:1]
        self._rest = temp[:, 1:]

        temp = tf.math.top_k(tf.gather(self._y, self._rest, axis=1, batch_dims=1), k=2)[0]
        self._second_margin = temp[:, 0] - temp[:, 1]

        self._l = tf.one_hot(tf.reshape(self._highest, -1), self._n_outputs)
        self._neg_l = tf.where(tf.equal(self._l, 0.), 1., 0.)

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def fit(self, X, y, alpha=1e-2, beta=1e-2, gamma=0., rho=1e-2, **kwargs):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._rho = rho
        return super().fit(X, y, **kwargs)
