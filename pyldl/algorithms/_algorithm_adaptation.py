import numpy as np
from sklearn.neighbors import NearestNeighbors

import keras
import tensorflow as tf

from pyldl.algorithms.utils import RProp
from pyldl.algorithms.base import BaseLDL, BaseDeepLDL, BaseGD, BaseAdam


EPS = np.finfo(np.float32).eps


class AA_KNN(BaseLDL):
    """:class:`AA-kNN <pyldl.algorithms.AA_KNN>` is proposed in paper :cite:`2016:geng`.
    """

    def fit(self, X, D, k=5):
        super().fit(X, D)
        self._model = NearestNeighbors(n_neighbors=k).fit(self._X)
        return self

    def predict(self, X):
        _, inds = self._model.kneighbors(X)
        return np.average(self._D[inds], axis=1)


class AA_BP(BaseGD, BaseDeepLDL):
    """:class:`AA-BP <pyldl.algorithms.AA_BP>` is proposed in paper :cite:`2016:geng`.
    """
    pass


class CPNN(BaseGD, BaseDeepLDL):
    """:class:`CPNN <pyldl.algorithms.CPNN>` is proposed in paper :cite:`2013:geng`.

    :term:`RProp` is used as the optimizer.
    """

    def __init__(self, mode='none', v=5, n_hidden=64, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        if mode in ['none', 'binary', 'augment']:
            self._mode = mode
        else:
            raise ValueError("The argument 'mode' can only be 'none', 'binary' or 'augment'.")
        self._v = v

    @staticmethod
    @tf.function
    def loss_function(D, D_pred):
        return tf.math.reduce_mean(keras.losses.kl_divergence(D, D_pred))

    def _get_default_model(self):
        extra = 1 if self._mode == 'none' else self._n_outputs
        return self.get_3layer_model(self._n_features + extra, self._n_hidden, 1, output_activation=None)

    def _get_default_optimizer(self):
        return RProp()

    def _before_train(self):
        if self._mode == 'augment':
            n = self._n_samples
            one_hot = tf.one_hot(tf.math.argmax(self._D, axis=1), self._n_outputs)
            self._X = tf.repeat(self._X, self._v, axis=0)
            self._D = tf.repeat(self._D, self._v, axis=0)
            one_hot = tf.repeat(one_hot, self._v, axis=0)
            v = tf.reshape(tf.tile([1 / (i + 1) for i in range(self._v)], [n]), (-1, 1))
            self._D += self._D * one_hot * v

    def _make_inputs(self, X):
        temp = tf.reshape(tf.tile([i + 1 for i in range(self._n_outputs)], [X.shape[0]]), (-1, 1))
        if self._mode != 'none':
            temp = tf.one_hot(tf.reshape(temp, (-1, )) - 1, depth=self._n_outputs)
        return tf.concat([tf.cast(tf.repeat(X, self._n_outputs, axis=0), dtype=tf.float32),
                          tf.cast(temp, dtype=tf.float32)], axis=1)

    def _call(self, X):
        inputs = self._make_inputs(X)
        outputs = self._model(inputs)
        results = tf.reshape(outputs, (X.shape[0], self._n_outputs))
        return keras.activations.softmax(results)


class BCPNN(CPNN):
    """:class:`BCPNN <pyldl.algorithms.BCPNN>` is proposed in paper :cite:`2017:yang`.

    :term:`RProp` is used as the optimizer.

    This algorithm is based on :class:`CPNN <pyldl.algorithms.CPNN>`. See also:

    .. bibliography:: bib/ldl/references.bib
        :filter: False
        :labelprefix: BCPNN-
        :keyprefix: bcpnn-

        2013:geng
    """

    def __init__(self, **params):
        super().__init__(mode='binary', **params)


class ACPNN(CPNN):
    """:class:`ACPNN <pyldl.algorithms.ACPNN>` is proposed in paper :cite:`2017:yang`.

    :term:`RProp` is used as the optimizer.

    This algorithm is based on :class:`CPNN <pyldl.algorithms.CPNN>`. See also:

    .. bibliography:: bib/ldl/references.bib
        :filter: False
        :labelprefix: ACPNN-
        :keyprefix: acpnn-

        2013:geng
    """

    def __init__(self, **params):
        super().__init__(mode='augment', **params)


class LDLF(BaseAdam, BaseDeepLDL):
    """:class:`LDLF <pyldl.algorithms.LDLF>` is proposed in paper :cite:`2017:shen`.

    :term:`Adam` is used as the optimizer.

    This algorithm employs deep neural decision forests. See also:

    .. bibliography:: bib/ldl/references.bib
        :filter: False
        :labelprefix: LDLF-
        :keyprefix: ldlf-

        2015:kontschieder
    """

    def __init__(self, n_estimators=5, n_depth=6, n_hidden=64, n_latent=64, random_state=None):
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

    def _get_default_model(self):
        return self.get_3layer_model(self._n_features, self._n_hidden, self._n_latent, output_activation='sigmoid')

    def _before_train(self):
        self._phi = [np.random.choice(
            np.arange(self._n_latent), size=self._n_leaves, replace=False
        ) for _ in range(self._n_estimators)]

        self._pi = [tf.Variable(
            initial_value = tf.constant_initializer(1 / self.n_outputs)(
                shape=[self._n_leaves, self._n_outputs]
            ),
            dtype="float32", trainable=True,
        ) for _ in range(self._n_estimators)]

    @tf.function
    def _loss(self, X, D, start=None, end=None):
        loss = 0.
        for i in range(self._n_estimators):
            _mu = self._call(X, i)
            _prob = tf.matmul(_mu, self._pi[i])

            loss += tf.math.reduce_mean(keras.losses.kl_divergence(D, _prob))

            _D = tf.expand_dims(D, axis=1)
            _pi = tf.expand_dims(self._pi[i], axis=0)
            _mu = tf.expand_dims(_mu, axis=2)
            _prob = tf.clip_by_value(
                tf.expand_dims(_prob, axis=1), clip_value_min=1e-7, clip_value_max=1.)
            _new_pi = tf.multiply(tf.multiply(_D, _pi), _mu) / _prob
            _new_pi = tf.reduce_sum(_new_pi, axis=0)
            _new_pi = keras.activations.softmax(_new_pi)
            self._pi[i].assign(_new_pi)

        loss /= self._n_estimators
        return loss

    def predict(self, X):
        res = np.zeros([X.shape[0], self._n_outputs], dtype=np.float32)
        for i in range(self._n_estimators):
            res += tf.matmul(self._call(X, i), self._pi[i])
        return res / self._n_estimators


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
