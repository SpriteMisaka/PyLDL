import keras
import tensorflow as tf

from pyldl.metrics import score
from pyldl.algorithms.base import BaseDeepLDL, BaseDeepLDLClassifier, DeepBFGS


class LDL4C(BaseDeepLDLClassifier, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)
        top2 = tf.gather(y_pred, self._top2, axis=1, batch_dims=1)
        margin = tf.reduce_sum(tf.maximum(0., 1. - (top2[:, 0] - top2[:, 1]) / self._rho))
        mae = keras.losses.mean_absolute_error(y, y_pred)
        return tf.reduce_sum(self._entropy * mae) + self._alpha * margin + self._beta * BaseDeepLDL._l2_reg(self._model)
    
    def fit(self, X, y, max_iterations=50, alpha=1e-2, beta=1e-6, rho=1e-2):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta
        self._rho = rho

        self._top2 = tf.math.top_k(self._y, k=2)[1]
        self._entropy = tf.cast(-tf.reduce_sum(self._y * tf.math.log(self._y) + 1e-7, axis=1), dtype=tf.float32)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])

        self._optimize_bfgs(self._model, self._loss, self._X, self._y, max_iterations=max_iterations)


class LDL_HR(BaseDeepLDLClassifier, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)
        
        highest = tf.gather(y_pred, self._highest, axis=1, batch_dims=1)
        rest = tf.gather(y_pred, self._rest, axis=1, batch_dims=1)
        margin = tf.reduce_sum(tf.maximum(0., 1. - (highest - rest) / self._rho))

        real_rest = tf.gather(y, self._rest, axis=1, batch_dims=1)
        rest_mae = tf.reduce_sum(keras.losses.mean_absolute_error(real_rest, rest))

        mae = tf.reduce_sum(keras.losses.mean_absolute_error(self._l, y_pred))

        return mae + self._alpha * margin + self._beta * rest_mae + self._gamma * BaseDeepLDL._l2_reg(self._model)
    
    def fit(self, X, y, max_iterations=50, alpha=1e-2, beta=1e-2, gamma=1e-6, rho=1e-2):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._rho = rho

        temp = tf.math.top_k(self._y, k=self._n_outputs)[1]
        self._highest = temp[:, 0:1]
        self._rest = temp[:, 1:]

        self._l = tf.one_hot(tf.reshape(self._highest, -1), self._n_outputs)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])

        self._optimize_bfgs(self._model, self._loss, self._X, self._y, max_iterations=max_iterations)


class LDLM(BaseDeepLDLClassifier):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)

        pred_margin = tf.reduce_sum(tf.clip_by_value(
            tf.reduce_sum(tf.abs(self._l - y_pred), axis=1) - self._rho,
            0., float('inf')))
        
        highest = tf.gather(y_pred, self._highest, axis=1, batch_dims=1)
        rest = tf.gather(y_pred, self._rest, axis=1, batch_dims=1)
        label_margin = tf.reduce_sum(tf.maximum(0., 1. - (highest - rest) / self._rho))

        second_margin = tf.reduce_sum(tf.clip_by_value(
            tf.reduce_sum(tf.abs((y - y_pred) * self._neg_l), axis=1) - self._second_margin,
            0., float('inf')))

        return pred_margin + self._alpha * label_margin + \
            self._beta * second_margin + self._gamma * BaseDeepLDL._l2_reg(self._model)

    def fit(self, X, y, learning_rate=5e-4, epochs=1000,
            alpha=1e-2, beta=1e-2, gamma=1e-6, rho=1e-2):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._rho = rho

        temp = tf.math.top_k(self._y, k=self._n_outputs)[1]
        self._highest = temp[:, 0:1]
        self._rest = temp[:, 1:]

        temp = tf.math.top_k(tf.gather(self._y, self._rest, axis=1, batch_dims=1), k=2)[0]
        self._second_margin = temp[:, 0] - temp[:, 1]

        self._l = tf.one_hot(tf.reshape(self._highest, -1), self._n_outputs)
        self._neg_l = tf.where(tf.equal(self._l, 0.), 1., 0.)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])
        
        self._optimizer = keras.optimizers.SGD(learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
