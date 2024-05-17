import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


class LDL_LRR(BaseBFGS, BaseDeepLDL):

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
    def _loss(self, X, y):
        y_pred = self._model(X)
        kl = tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))
        rank = LDL_LRR.ranking_loss(y_pred, self._P, self._W) / (2 * X.shape[0])
        return kl + self._alpha * rank + self._beta * BaseDeepLDL._l2_reg(self._model)

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _before_train(self):
        self._P, self._W = LDL_LRR.preprocessing(self._y)

    def fit(self, X, y, alpha=1e-2, beta=1e-8, max_iterations=50):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, max_iterations=max_iterations)
