import numpy as np
from sklearn.neighbors import NearestNeighbors

import keras
import tensorflow as tf

from pyldl.algorithms.utils import RProp
from pyldl.algorithms.base import BaseLDL, BaseDeepLDL, BaseGD
from pyldl.algorithms.loss_function_engineering import _CAD, _QFD2, _CJS


class AA_KNN(BaseLDL):
    """:class:`AA-kNN <pyldl.algorithms.AA_KNN>` is proposed in paper :cite:`2016:geng`.
    """

    def fit(self, X, y, k=5):
        super().fit(X, y)
        self._model = NearestNeighbors(n_neighbors=k).fit(self._X)
        return self

    def predict(self, X):
        _, inds = self._model.kneighbors(X)
        return np.average(self._y[inds], axis=1)


class AA_BP(BaseGD, BaseDeepLDL):
    """:class:`AA-BP <pyldl.algorithms.AA_BP>` is proposed in paper :cite:`2016:geng`.
    """
    pass


class CAD(_CAD, AA_BP):
    """:class:`CAD <pyldl.algorithms.CAD>` is proposed in paper :cite:`2023:wen`.
    """
    pass


class QFD2(_QFD2, AA_BP):
    """:class:`QFD2 <pyldl.algorithms.QFD2>` is proposed in paper :cite:`2023:wen`.
    """
    pass


class CJS(_CJS, AA_BP):
    """:class:`CJS <pyldl.algorithms.CJS>` is proposed in paper :cite:`2023:wen`.
    """
    pass


class CPNN(BaseGD, BaseDeepLDL):
    """:class:`CPNN <pyldl.algorithms.CPNN>` is proposed in paper :cite:`2013:geng`.
    """

    def _not_proper_mode(self):
        raise ValueError("The argument 'mode' can only be 'none', 'binary' or 'augment'.")

    def __init__(self, mode='none', v=5, n_hidden=64, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        if mode == 'none' or mode == 'binary' or mode == 'augment':
            self._mode = mode
        else:
            self._not_proper_mode()
        self._v = v

    @staticmethod
    @tf.function
    def loss_function(y, y_pred):
        return tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))

    def _get_default_model(self):
        input_shape = (self._n_features + (1 if self._mode == 'none' else self._n_outputs),)
        return keras.Sequential([keras.layers.InputLayer(input_shape=input_shape),
                                 keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                 keras.layers.Dense(1, activation=None)])

    def _get_default_optimizer(self):
        return RProp()

    def _before_train(self):
        if self._mode == 'augment':
            n = self._X.shape[0]
            one_hot = tf.one_hot(tf.math.argmax(self._y, axis=1), self.n_outputs)
            self._X = tf.repeat(self._X, self._v, axis=0)
            self._y = tf.repeat(self._y, self._v, axis=0)
            one_hot = tf.repeat(one_hot, self._v, axis=0)
            v = tf.reshape(tf.tile([1 / (i + 1) for i in range(self._v)], [n]), (-1, 1))
            self._y += self._y * one_hot * v

    def _make_inputs(self, X):
        temp = tf.reshape(tf.tile([i + 1 for i in range(self._n_outputs)], [X.shape[0]]), (-1, 1))
        if self._mode != 'none':
            temp = tf.one_hot(tf.reshape(temp, (-1, )) - 1, depth=self._n_outputs)
        return tf.concat([tf.cast(tf.repeat(X, self._n_outputs, axis=0), dtype=tf.float32),
                          tf.cast(temp, dtype=tf.float32)],
                          axis=1)

    def _call(self, X):
        inputs = self._make_inputs(X)
        outputs = self._model(inputs)
        results = tf.reshape(outputs, (X.shape[0], self._n_outputs))
        b = tf.reshape(-tf.math.log(tf.math.reduce_sum(tf.math.exp(results), axis=1)), (-1, 1))
        return tf.math.exp(b + results)


class BCPNN(CPNN):
    """:class:`BCPNN <pyldl.algorithms.BCPNN>` is proposed in paper :cite:`2017:yang`.

    :class:`BCPNN <pyldl.algorithms.BCPNN>` is based on :class:`CPNN <pyldl.algorithms.CPNN>`. See also:

    .. bibliography:: ldl_references.bib
        :filter: False
        :labelprefix: BCPNN-
        :keyprefix: bcpnn-

        2013:geng
    """

    def __init__(self, **params):
        super().__init__(mode='binary', **params)


class ACPNN(CPNN):
    """:class:`ACPNN <pyldl.algorithms.ACPNN>` is proposed in paper :cite:`2017:yang`.

    :class:`ACPNN <pyldl.algorithms.ACPNN>` is based on :class:`CPNN <pyldl.algorithms.CPNN>`. See also:

    .. bibliography:: ldl_references.bib
        :filter: False
        :labelprefix: ACPNN-
        :keyprefix: acpnn-

        2013:geng
    """

    def __init__(self, **params):
        super().__init__(mode='augment', **params)
