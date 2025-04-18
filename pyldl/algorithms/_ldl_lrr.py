import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


EPS = np.finfo(np.float32).eps


class LDL_LRR(BaseBFGS, BaseDeepLDL):
    """:class:`LDL-LRR <pyldl.algorithms.LDL_LRR>` is proposed in paper :cite:`2023:jia`.

    :term:`BFGS` is used as the optimization algorithm.
    """

    def __init__(self, alpha=1e-2, beta=0., **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._beta = beta

    @staticmethod
    @tf.function
    def ranking_loss(D_pred, P, W):
        P_hat = tf.math.sigmoid(D_pred[:, :, None] - D_pred[:, None, :])
        l = ((1 - P) * tf.math.log(tf.clip_by_value(1 - P_hat, EPS, 1.)) +\
              P * tf.math.log(tf.clip_by_value(P_hat, EPS, 1.))) * W
        return -tf.reduce_sum(l)

    @staticmethod
    @tf.function
    def preprocessing(D):
        diff = D[:, :, None] - D[:, None, :]
        return tf.where(diff > .5, 1., 0.), tf.square(diff)

    @tf.function
    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        D_pred = keras.activations.softmax(self._X @ theta)
        kld = tf.math.reduce_mean(keras.losses.kl_divergence(self._D, D_pred))
        rnk = self.ranking_loss(D_pred, self._P, self._W) / (2 * self._n_samples)
        return kld + self._alpha * rnk + self._beta * self._l2_reg(theta)

    def _before_train(self):
        self._P, self._W = LDL_LRR.preprocessing(self._D)
