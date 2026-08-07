import numpy as np

import keras
import keras.ops as ops

from sklearn.neighbors import NearestNeighbors

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam
from pyldl.algorithms.utils import pairwise_pearsonr


EPS = np.finfo(np.float32).eps


class LDL_HVLC(BaseAdam, BaseDeepLDL):
    """:class:`LDL-HVLC <pyldl.algorithms.LDL_HVLC>` is proposed in paper :cite:`2024:lin`. 
    HVLC refers to *horizontal & vertical label correlation*.
    """

    def __init__(self, k=5, alpha=1e-3, beta=1e-3, gamma=1e-5, delta=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs, activation=None)

    def _loss(self, X, D, start, end):
        D_pred = keras.activations.softmax(self._model(X) + self._C[start:end] @ self._M)
        temp = ops.sum(ops.square(D_pred - self._C[start:end]), axis=1)
        horizontal = ops.sum(self._p[start:end] * temp)
        diff = ops.expand_dims(D_pred, axis=2) - ops.expand_dims(D_pred, axis=1)
        vertical = ops.sum(ops.expand_dims(self._P, axis=0) * diff**2)
        kl = ops.sum(keras.losses.kl_divergence(D, D_pred))
        hv = self.alpha * horizontal + self.beta * vertical
        reg = self.gamma * self._l2_reg(self._model.trainable_variables) + self.delta * self._l2_reg(self._M)
        return kl + hv + reg

    def _construct_C(self, X, self_include=True):
        _, inds = self._knn.kneighbors(self._to_numpy(X))
        return ops.mean(ops.take(self._D, inds[:, :-1] if self_include else inds[:, 1:], axis=0), axis=1)

    def _before_train(self):
        self._M = self.add_weight(
            name='M', shape=(self._n_outputs, self._n_outputs),
            initializer=keras.initializers.RandomNormal(), trainable=True
        )
        self._knn = NearestNeighbors(n_neighbors=self.k+1).fit(self._to_numpy(self._X))
        self._C = self._construct_C(self._X, self_include=False)
        self._p = ops.convert_to_tensor([pairwise_pearsonr(self._C[i], self._D[i]) for i in range(self._n_samples)], dtype="float32")
        self._P = ops.convert_to_tensor(pairwise_pearsonr(ops.transpose(self._D)), dtype="float32")

    def train_step(self, _, *args, **kwargs):
        super().train_step([self._model.trainable_variables, self._optimizer], *args, **kwargs)
        super().train_step([[self._M], self._M_optimizer], *args, **kwargs)

    def fit(self, X, D, *, M_optimizer=None, **kwargs):
        self._M_optimizer = M_optimizer or self._get_default_optimizer()
        return super().fit(X, D, **kwargs)

    def predict(self, X):
        return self._to_numpy(keras.activations.softmax(self._model(X) + self._construct_C(X) @ self._M))
