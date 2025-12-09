import numpy as np
import tensorflow as tf
 
from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class SNEFY_LDL(BaseAdam, BaseDeepLDL):
    """:class:`SNEFY-LDL <pyldl.algorithms.SNEFY_LDL>` is proposed in paper :cite:`2025:zhang`. 
    SNEFY refers to *squared neural family*.
    """

    def __init__(self, n_hidden=64, n_latent=32, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)

    @tf.function
    def _kernel(self, features):
        gamma = lambda x: tf.math.exp(tf.math.lgamma(x))
        temp = self._b + features
        F = tf.expand_dims(temp, axis=1) + tf.expand_dims(temp, axis=2)
        W = tf.expand_dims(self._W, axis=1) + tf.expand_dims(self._W, axis=2)
        numerator = tf.reduce_prod(gamma(W + 1), axis=0)
        denominator = gamma(tf.cast(self._n_outputs, tf.float32) + tf.reduce_sum(W, axis=0))
        return tf.reshape(tf.math.exp(F) * (numerator / denominator), (-1, self._n_hidden, self._n_hidden))

    @tf.function
    def _calculate_VKV(self, features):
        K = self._kernel(features)
        VTV = tf.transpose(self._V) @ self._V
        return K * VTV

    @tf.function
    def _loss(self, X, D, start, end):
        features = self._encoder(X)
        latent = tf.math.exp(self._log_D[start:end] @ self._W + features + self._b)
        net = tf.reshape((tf.norm(latent @ tf.transpose(self._V), axis=1)**2), (-1, ))
        log = tf.math.log(net + EPS)
        VKV = self._calculate_VKV(features)
        return - tf.reduce_mean(log - tf.math.log(tf.reduce_sum(VKV, axis=(1, 2)) + EPS))

    def _before_train(self):
        self._log_D = tf.math.log(self._D)
        self._encoder = self.get_3layer_model(self._n_features, self._n_hidden, self._n_hidden,
                                              hidden_activation='relu', output_activation=None)
        self._W = tf.Variable(tf.random.normal((self._n_outputs, self._n_hidden)), trainable=True,
                              constraint=lambda x: tf.maximum(x, -.495))
        self._V = tf.Variable(tf.random.normal((self._n_latent, self._n_hidden), 0., 1.) *\
                              tf.sqrt(1. / (self._n_latent * self._n_hidden)), trainable=True)
        self._b = tf.Variable(tf.zeros((1, self._n_hidden)), trainable=True)

    def fit(self, X, Y, *, batch_size=64, **kwargs):
        return super().fit(X, Y, batch_size=batch_size, **kwargs)

    def predict(self, X, return_uncertainty=False):
        features = self._encoder(X)
        VKV = self._calculate_VKV(features)
        W = tf.expand_dims(self._W, axis=1) + tf.expand_dims(self._W, axis=2)
        alpha = W + 1
        alpha_0 = self._n_outputs + tf.reduce_sum(W, axis=0)
        E1 = alpha / alpha_0
        E2 = (alpha * (alpha + 1)) / (alpha_0 * (alpha_0 + 1))
        tempE1 = tf.einsum('bmn,dmn->bd', VKV, E1)
        tempE2 = tf.einsum('bmn,dmn->bd', VKV, E2)
        VKV_sum = tf.reshape(tf.reduce_sum(VKV, axis=(1, 2)), (-1, 1))
        D_pred = (tempE1 / VKV_sum).numpy()
        if return_uncertainty:
            uncertainty = (tempE2 / VKV_sum).numpy() - D_pred**2
            return D_pred, uncertainty
        return D_pred
