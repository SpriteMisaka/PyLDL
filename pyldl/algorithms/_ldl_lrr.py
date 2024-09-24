import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


EPS = np.finfo(np.float32).eps


class LDL_LRR(BaseBFGS, BaseDeepLDL):
    """:class:`LDL-LRR <pyldl.algorithms.LDL_LRR>` is proposed in paper :cite:`2023:jia`.

    :term:`BFGS` is used as the optimization algorithm.
    """

    @staticmethod
    @tf.function
    def ranking_loss(y_pred, P, W):
        Phat = tf.math.sigmoid(y_pred[:, :, None] - y_pred[:, None, :])
        l = ((1 - P) * tf.math.log(tf.clip_by_value(1 - Phat, EPS, 1.)) +\
              P * tf.math.log(tf.clip_by_value(Phat, EPS, 1.))) * W
        return -tf.reduce_sum(l)

    @staticmethod
    @tf.function
    def preprocessing(y):
        diff = y[:, :, None] - y[:, None, :]
        return tf.where(diff > .5, 1., 0.), tf.square(diff)

    @tf.function
    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        y_pred = keras.activations.softmax(self._X @ theta)
        kld = tf.math.reduce_mean(keras.losses.kl_divergence(self._y, y_pred))
        rnk = self.ranking_loss(y_pred, self._P, self._W) / (2 * self._X.shape[0])
        return kld + self._alpha * rnk + self._beta * self._l2_reg(theta)

    def _before_train(self):
        self._P, self._W = self.preprocessing(self._y)

    def fit(self, X, y, alpha=1e-2, beta=0., **kwargs):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, **kwargs)
