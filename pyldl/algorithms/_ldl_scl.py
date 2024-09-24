import numpy as np
from sklearn.svm import SVR
from sklearn.cluster import KMeans

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


class LDL_SCL(BaseAdam, BaseDeepLDL):
    """:class:`LDL-SCL <pyldl.algorithms.LDL_SCL>` is proposed in paper :cite:`2018:zheng`.

    :term:`Adam` is used as optimizer.

    See also:

    .. bibliography:: ldl_references.bib
        :filter: False
        :labelprefix: LDL-SCL-
        :keyprefix: ldl-scl-

        2021:jia
    """

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _before_train(self):
        self._P = tf.cast(KMeans(n_clusters=self._n_clusters).fit(self._y).cluster_centers_,
                          dtype=tf.float32)
        self._C = tf.Variable(tf.zeros((self._X.shape[0], self._n_clusters)) + 1e-6,
                              trainable=True)
        self._W = tf.Variable(tf.random.normal((self._n_clusters, self._n_outputs)),
                              trainable=True)

    @tf.function
    def _loss(self, X, y, start, end):
        y_pred = keras.activations.softmax(self._model(X) + tf.matmul(self._C[start:end], self._W))

        kl = tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))
        corr = tf.math.reduce_mean(self._C[start:end] * keras.losses.mean_squared_error(
            tf.expand_dims(y_pred, 1), tf.expand_dims(self._P, 0)
        ))
        barr = tf.math.reduce_mean(1 / self._C[start:end])

        return kl + self._alpha * corr + self._beta * barr

    def fit(self, X, y, n_clusters=5, alpha=1e-3, beta=1e-6, **kwargs):
        self._n_clusters = n_clusters
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, **kwargs)

    def predict(self, X):

        C = np.zeros((X.shape[0], self._n_clusters))
        for i in range(self._n_clusters):
            lr = SVR()
            lr.fit(self._X.numpy(), self._C.numpy()[:, i].reshape(-1))
            C[:, i] = lr.predict(X).reshape(1, -1)
        C = tf.cast(C, dtype=tf.float32)

        return keras.activations.softmax(self._model(X) + tf.matmul(C, self._W))
