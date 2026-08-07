import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam
from pyldl.algorithms.utils import log_gamma


EPS = np.finfo(np.float32).eps


@keras.saving.register_keras_serializable()
class SNEFY_LDL(BaseAdam, BaseDeepLDL):
    """:class:`SNEFY-LDL <pyldl.algorithms.SNEFY_LDL>` is proposed in paper :cite:`2025:zhang`.
    SNEFY refers to *squared neural family*.
    """

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    def _kernel(self, features):
        temp = self._b + features
        F = ops.expand_dims(temp, axis=1) + ops.expand_dims(temp, axis=2)
        W = ops.expand_dims(self._W, axis=1) + ops.expand_dims(self._W, axis=2)
        log_numerator = ops.sum(log_gamma(W + 1), axis=0)
        log_denominator = log_gamma(
            ops.cast(self._n_outputs, "float32") + ops.sum(W, axis=0)
        )
        return ops.reshape(
            ops.exp(F + log_numerator - log_denominator),
            (-1, self._n_hidden, self._n_hidden)
        )

    def _calculate_VKV(self, features):
        K = self._kernel(features)
        VTV = ops.transpose(self._V) @ self._V
        return K * VTV

    def _loss(self, X, D, start, end):
        features = self._model(X)
        latent = ops.exp(self._log_D[start:end] @ self._W + features + self._b)
        net = ops.reshape(ops.norm(latent @ ops.transpose(self._V), axis=1)**2, (-1,))
        log = ops.log(net + EPS)
        VKV = self._calculate_VKV(features)
        return -ops.mean(log - ops.log(ops.sum(VKV, axis=(1, 2)) + EPS))

    def _get_default_model(self):
        return self.get_3layer_model(self._n_features, self._n_hidden, self._n_hidden,
                                     hidden_activation='relu', output_activation=None)

    def _before_train(self):
        self._log_D = ops.log(self._D)
        self._W = self.add_weight(
            name='W', shape=(self._n_outputs, self._n_hidden),
            initializer=keras.initializers.RandomNormal(
                stddev=np.sqrt(2. / self._n_outputs)
            ), trainable=True
        )
        self._W.assign(ops.maximum(self._W, -.495))
        self._V = self.add_weight(
            name='V', shape=(self._n_latent, self._n_hidden),
            initializer=keras.initializers.RandomNormal(
                stddev=np.sqrt(1. / (self._n_latent * self._n_hidden))
            ), trainable=True
        )
        self._b = self.add_weight(
            name='b', shape=(1, self._n_hidden),
            initializer=keras.initializers.Zeros(), trainable=True
        )

    def train_step(self, *args, **kwargs):
        super().train_step(*args, **kwargs)
        self._W.assign(ops.maximum(self._W, -.495))

    def predict(self, X, return_uncertainty=False):
        features = self._model(X)
        VKV = self._calculate_VKV(features)
        W = ops.expand_dims(self._W, axis=1) + ops.expand_dims(self._W, axis=2)
        alpha = W + 1
        alpha_0 = self._n_outputs + ops.sum(W, axis=0)
        E1 = alpha / alpha_0
        E2 = (alpha * (alpha + 1)) / (alpha_0 * (alpha_0 + 1))
        tempE1 = ops.einsum('bmn,dmn->bd', VKV, E1)
        tempE2 = ops.einsum('bmn,dmn->bd', VKV, E2)
        VKV_sum = ops.reshape(ops.sum(VKV, axis=(1, 2)), (-1, 1))
        D_pred = self._to_numpy(tempE1 / VKV_sum)
        if return_uncertainty:
            uncertainty = self._to_numpy(tempE2 / VKV_sum) - D_pred**2
            return D_pred, uncertainty
        return D_pred
