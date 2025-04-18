import numpy as np

import keras
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors

from pyldl.algorithms.base import BaseDeepLDL, BaseGD
from pyldl.algorithms.utils import pairwise_pearsonr


EPS = np.finfo(np.float32).eps


class LDL_HVLC(BaseGD, BaseDeepLDL):
    """:class:`LDL-HVLC <pyldl.algorithms.LDL_HVLC>` is proposed in paper :cite:`2024:lin`.
    """

    def __init__(self, k=5, alpha=1e-3, beta=1e-3, gamma=1e-5, delta=1e-5, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs, activation=None)

    def _get_default_optimizer(self):
        return keras.optimizers.SGD(1e-2, momentum=.9)

    @tf.function
    def _loss(self, X, D, start, end):
        D_pred = keras.activations.softmax(self._model(X) + self._C[start:end] @ self._M)
        temp = tf.reduce_sum(tf.square(D_pred - self._C[start:end]), axis=1)
        horizontal = tf.reduce_sum(self._p[start:end] * temp)
        diff = tf.expand_dims(D_pred, axis=2) - tf.expand_dims(D_pred, axis=1)
        vertical = tf.reduce_sum(tf.expand_dims(self._P, axis=0) * diff**2)
        kl = tf.reduce_sum(keras.losses.kl_divergence(D, D_pred))
        hv = self._alpha * horizontal + self._beta * vertical
        reg = self._gamma * self._l2_reg(self._model.trainable_variables) + self._delta * self._l2_reg(self._M)
        return kl + hv + reg

    def _construct_C(self, X, self_include=True):
        _, inds = self._knn.kneighbors(X)
        return tf.reduce_mean(tf.gather(self._D, inds[:, :-1] if self_include else inds[:, 1:], axis=0), axis=1)

    def _before_train(self):
        self._M = tf.Variable(tf.random.normal((self._n_outputs, self._n_outputs)), trainable=True)
        self._knn = NearestNeighbors(n_neighbors=self._k+1).fit(self._X)
        self._C = self._construct_C(self._X, self_include=False)
        self._p = tf.convert_to_tensor([pairwise_pearsonr(self._C[i], self._D[i]) for i in range(self._n_samples)], dtype=tf.float32)
        self._P = tf.convert_to_tensor(pairwise_pearsonr(tf.transpose(self._D)) , dtype=tf.float32)

    def train_step(self, batch, loss, _, start, end):
        super().train_step(batch, loss, self._model.trainable_variables, start, end)
        super().train_step(batch, loss, [self._M], start, end)

    def predict(self, X):
        return keras.activations.softmax(self._model(X) + self._construct_C(X) @ self._M)
