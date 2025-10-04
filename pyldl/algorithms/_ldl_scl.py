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

    .. bibliography:: bib/ldl/references.bib
        :filter: False
        :labelprefix: LDL-SCL-
        :keyprefix: ldl-scl-

        2021:jia
    """

    def __init__(self, n_clusters=5, alpha=1e-3, beta=1e-6, **kwargs):
        super().__init__(**kwargs)
        self._n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _before_train(self):
        self._P = tf.cast(KMeans(n_clusters=self._n_clusters).fit(self._D).cluster_centers_,
                          dtype=tf.float32)
        self._C = tf.Variable(tf.zeros((self._n_samples, self._n_clusters)) + 1e-6,
                              trainable=True)
        self._W = tf.Variable(tf.random.normal((self._n_clusters, self._n_outputs)),
                              trainable=True)

    @staticmethod
    @tf.function
    def scl_loss(D_pred, P, C):
        corr = tf.math.reduce_mean(C * keras.losses.mean_squared_error(
            tf.expand_dims(D_pred, 1), tf.expand_dims(P, 0)
        ))
        barr = tf.math.reduce_mean(1 / C)
        return corr, barr

    @tf.function
    def _loss(self, X, D, start, end):
        D_pred = keras.activations.softmax(self._model(X) + tf.matmul(self._C[start:end], self._W))
        kl = tf.math.reduce_mean(keras.losses.kl_divergence(D, D_pred))
        corr, barr = self.scl_loss(D_pred, self._P, self._C[start:end])
        return kl + self.alpha * corr + self.beta * barr

    @staticmethod
    def construct_C(X, old_X, old_C):
        C = np.zeros((X.shape[0], old_C.shape[1]))
        for i in range(old_C.shape[1]):
            lr = SVR()
            lr.fit(old_X, old_C.numpy()[:, i].reshape(-1))
            C[:, i] = lr.predict(X).reshape(1, -1)
        return tf.cast(C, dtype=tf.float32)

    def predict(self, X):
        C = self.construct_C(X, self._X, self._C)
        return keras.activations.softmax(self._model(X) + tf.matmul(C, self._W))

    @property
    def n_clusters(self):
        return self._n_clusters
