import tensorflow as tf
import tensorflow_addons as tfa

from pyldl.algorithms.base import BaseDeepLDL, BaseGD


class IncomLDL(BaseGD, BaseDeepLDL):

    def _loss(self, X, y):
        y_pred = self._model(X)
        trace_norm = 0.
        for i in self._model.trainable_variables:
            trace_norm += tf.linalg.trace(tf.sqrt(tf.matmul(tf.transpose(i), i)))
        fro_norm = tf.reduce_sum(tf.square(self._mask * (y_pred - y)))
        return fro_norm / 2. + self._alpha * trace_norm

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _get_default_optimizer(self):
        return tfa.optimizers.ProximalAdagrad()

    def fit(self, X, y, mask, alpha=1e-3, **kwargs):
        self._alpha = alpha
        self._mask = tf.where(mask, 0., 1.)
        return super().fit(X, y, **kwargs)
