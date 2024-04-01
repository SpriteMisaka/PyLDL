import numpy as np
from sklearn.svm import SVR
from sklearn.cluster import KMeans

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL


class LDL_SCL(BaseDeepLDL):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = keras.activations.softmax(self._model(X) + tf.matmul(self._C, self._W))

        kl = tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))

        corr = tf.math.reduce_mean(self._C * keras.losses.mean_squared_error(
            tf.expand_dims(y_pred, 1), tf.expand_dims(self._P, 0)
        ))

        barr = tf.math.reduce_mean(1 / self._C)

        return kl + self._alpha * corr + self._beta * barr

    def fit(self, X, y, n_clusters=5, learning_rate=5e-2, epochs=3000, alpha=1e-3, beta=1e-6):
        super().fit(X, y)

        self._n_clusters = n_clusters
        self._alpha = alpha
        self._beta = beta

        self._P = tf.cast(KMeans(n_clusters=self._n_clusters).fit(self._y).cluster_centers_,
                          dtype=tf.float32)

        self._C = tf.Variable(tf.zeros((self._X.shape[0], self._n_clusters)) + 1e-6,
                              trainable=True)

        self._W = tf.Variable(tf.random.normal((self._n_clusters, self._n_outputs)),
                              trainable=True)

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                        keras.layers.Dense(self._n_outputs, activation=None)])
        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    def predict(self, X):

        C = np.zeros((X.shape[0], self._n_clusters))
        for i in range(self._n_clusters):
            lr = SVR()
            lr.fit(self._X.numpy(), self._C.numpy()[:, i].reshape(-1))
            C[:, i] = lr.predict(X).reshape(1, -1)
        C = tf.cast(C, dtype=tf.float32)

        return keras.activations.softmax(self._model(X) + tf.matmul(C, self._W))
