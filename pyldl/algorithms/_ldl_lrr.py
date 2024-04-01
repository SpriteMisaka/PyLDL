import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, DeepBFGS


class LDL_LRR(BaseDeepLDL, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

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

    def fit(self, X, y, alpha=1e-2, beta=1e-8, max_iterations=50):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta

        self._P, self._W = LDL_LRR.preprocessing(self._y)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
            keras.layers.Dense(self._n_outputs, activation="softmax")])

        self._optimize_bfgs(self._model, self._loss, self._X, self._y, max_iterations)

    def predict(self, X):
        return self._model(X).numpy()
