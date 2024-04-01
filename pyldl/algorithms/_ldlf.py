import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL


class LDLF(BaseDeepLDL):

    def __init__(self, n_estimators=5, n_depth=6, n_hidden=None, n_latent=64, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        self._n_estimators = n_estimators
        self._n_depth = n_depth
        self._n_leaves = 2 ** n_depth

    def _call(self, X, i):
        decisions = tf.gather(self._model(X), self._phi[i], axis=1)
        decisions = tf.expand_dims(decisions, axis=2)
        decisions = tf.concat([decisions, 1 - decisions], axis=2)
        mu = tf.ones([X.shape[0], 1, 1])

        begin_idx = 1
        end_idx = 2

        for level in range(self._n_depth):
            mu = tf.reshape(mu, [X.shape[0], -1, 1])
            mu = tf.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [X.shape[0], self._n_leaves])

        return mu

    def fit(self, X, y, learning_rate=5e-2, epochs=3000):
        super().fit(X, y)

        self._phi = [np.random.choice(
            np.arange(self._n_latent), size=self._n_leaves, replace=False
        ) for _ in range(self._n_estimators)]

        self._pi = [tf.Variable(
            initial_value = tf.constant_initializer(1 / self.n_outputs)(
                shape=[self._n_leaves, self._n_outputs]
            ),
            dtype="float32", trainable=True,
        ) for _ in range(self._n_estimators)]

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                        keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                        keras.layers.Dense(self._n_latent, activation="sigmoid")])
        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as model_tape:
                loss = 0.
                for i in range(self._n_estimators):
                    _mu = self._call(X, i)
                    _prob = tf.matmul(_mu, self._pi[i])

                    loss += tf.math.reduce_mean(keras.losses.kl_divergence(self._y, _prob))

                    _y = tf.expand_dims(self._y, axis=1)
                    _pi = tf.expand_dims(self._pi[i], axis=0)
                    _mu = tf.expand_dims(_mu, axis=2)
                    _prob = tf.clip_by_value(
                        tf.expand_dims(_prob, axis=1), clip_value_min=1e-6, clip_value_max=1.0)
                    _new_pi = tf.multiply(tf.multiply(_y, _pi), _mu) / _prob
                    _new_pi = tf.reduce_sum(_new_pi, axis=0)
                    _new_pi = keras.activations.softmax(_new_pi)
                    self._pi[i].assign(_new_pi)

                loss /= self._n_estimators

            gradients = model_tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    def predict(self, X):
        res = np.zeros([X.shape[0], self._n_outputs], dtype=np.float32)
        for i in range(self._n_estimators):
            res += tf.matmul(self._call(X, i), self._pi[i])
        return res / self._n_estimators
