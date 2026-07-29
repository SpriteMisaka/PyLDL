import keras
import tensorflow as tf

import numpy as np

from pyldl.algorithms.base.deep import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


@keras.saving.register_keras_serializable()
class Duo_LDL(BaseAdam, BaseDeepLDL):
    """:class:`Duo-LDL <pyldl.algorithms.Duo_LDL>` is proposed in paper :cite:`2021:zychowski`.
    """

    def __init__(self, n_hidden=40, **kwargs):
        super().__init__(n_hidden, **kwargs)

    def _get_default_model(self):
        output_units = self._n_outputs * (self._n_outputs - 1)
        return self.get_3layer_model(self._n_features, self._n_hidden, output_units, output_activation='tanh')

    @tf.function
    def _loss(self, X, _, start, end):
        C_pred = self._call(X)
        return self.loss_function(self._C[start:end], C_pred)

    def _before_train(self):
        self._C = tf.concat([self._D - tf.roll(self._D, i, axis=1) for i in range(1, self._n_outputs)], axis=1)

    def predict(self, X):
        C_pred = self._call(X)
        shape = (X.shape[0], self._n_outputs - 1, self._n_outputs)
        C_pred_reshaped = tf.transpose(tf.reshape(C_pred, shape), (0, 2, 1))
        D_pred = ((tf.reduce_sum(C_pred_reshaped, axis=2) + 1) / self._n_outputs).numpy()
        return D_pred / (np.sum(D_pred, axis=1, keepdims=True) + EPS)

    def fit(self, X, Y, *, batch_size=50, **kwargs):
        return super().fit(X, Y, batch_size=batch_size, **kwargs)
