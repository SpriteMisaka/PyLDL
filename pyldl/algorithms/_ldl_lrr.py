import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


class LDL_LRR(BaseBFGS, BaseDeepLDL):
    """LDL-LRR is proposed in paper Label Distribution Learning by Maintaining Label Ranking Relation.
    """

    @staticmethod
    @tf.function
    def ranking_loss(y_pred, P, W):
        Phat = tf.math.sigmoid((tf.expand_dims(y_pred, -1) - tf.expand_dims(y_pred, 1)) * 100)
        l = ((1 - P) * tf.math.log(tf.clip_by_value(1 - Phat, 1e-9, 1.0)) + \
              P * tf.math.log(tf.clip_by_value(Phat, 1e-9, 1.0))) * W
        return -tf.reduce_sum(l)

    @staticmethod
    @tf.function
    def preprocessing(y):
        P = tf.where(tf.nn.sigmoid(tf.expand_dims(y, -1) - tf.expand_dims(y, 1)) > .5, 1., 0.)
        W = tf.square(tf.expand_dims(y, -1) - tf.expand_dims(y, 1))
        return P, W

    @tf.function
    def _loss(self, params_1d):
        y_pred = keras.activations.softmax(self._X @ self._params2model(params_1d)[0])
        kld = tf.math.reduce_mean(keras.losses.kl_divergence(self._y, y_pred))
        rnk = self.ranking_loss(y_pred, self._P, self._W) / (2 * self._X.shape[0])
        return kld + self._alpha * rnk + self._beta * self._l2_reg(self._model)

    def _before_train(self):
        self._P, self._W = self.preprocessing(self._y)

    def fit(self, X, y, alpha=1e-2, beta=1e-6, **kwargs):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, **kwargs)
