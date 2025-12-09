import keras
import tensorflow as tf

from tensorflow_probability.python.stats import correlation

from pyldl.algorithms.utils import pairwise_euclidean, non_diagonal
from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


class LDLLC(BaseBFGS, BaseDeepLDL):
    """:class:`LDLLC <pyldl.algorithms.LDLLC>` is proposed in paper :cite:`2018:jia`.

    :term:`BFGS` is used as optimization algorithm.
    """

    def __init__(self, alpha=1e-3, beta=0., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    @tf.function
    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        D_pred = keras.activations.softmax(self._X @ theta)
        kld = tf.reduce_sum(keras.losses.kl_divergence(self._D, D_pred))
        lc = tf.reduce_sum(non_diagonal(
            tf.sign(correlation(theta)) * pairwise_euclidean(tf.transpose(theta))
        )) / 2.
        return kld + self.alpha * lc + self.beta * self._l2_reg(theta)
