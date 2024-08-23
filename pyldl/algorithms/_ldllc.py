import numpy as np

import keras
import tensorflow as tf
import tensorflow_probability as tfp

from pyldl.algorithms.utils import pairwise_euclidean
from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


EPS = np.finfo(np.float32).eps


class LDLLC(BaseBFGS, BaseDeepLDL):
    """LDLLC is proposed in paper *Label Distribution Learning by Exploiting Label Correlations*.
    """

    @tf.function
    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        y_pred = keras.activations.softmax(self._X @ theta)
        kld = tf.reduce_sum(keras.losses.kl_divergence(self._y, y_pred))
        lc = tfp.stats.correlation(theta) * pairwise_euclidean(tf.transpose(theta))
        eye = tf.eye(self._n_outputs, dtype=tf.float32)
        lc = tf.reduce_sum(lc * (1 - eye)) / 2.
        return kld + self._alpha * lc + self._beta * self._l2_reg(theta)

    def fit(self, X, y, alpha=1e-3, beta=0., **kwargs):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, **kwargs)
