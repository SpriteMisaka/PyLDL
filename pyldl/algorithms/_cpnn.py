import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseGD, BaseDeepLDL


@keras.saving.register_keras_serializable()
class CPNN(BaseGD, BaseDeepLDL):
    """:class:`CPNN <pyldl.algorithms.CPNN>` is proposed in paper :cite:`2013:geng`.

    :term:`RProp` is used as the optimizer.
    """

    def __init__(self, mode='none', v=5, n_hidden=64, n_latent=None, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)
        if mode in ['none', 'binary', 'augment']:
            self._mode = mode
        else:
            raise ValueError("The argument 'mode' can only be 'none', 'binary' or 'augment'.")
        self._v = v

    @staticmethod
    def loss_function(D, D_pred):
        return ops.mean(keras.losses.kl_divergence(D, D_pred))

    def _get_default_model(self):
        extra = 1 if self._mode == 'none' else self._n_outputs
        return self.get_3layer_model(self._n_features + extra, self._n_hidden, 1, output_activation=None)

    def _get_default_optimizer(self):
        from pyldl.algorithms.optimizers import RProp
        return RProp()

    def _before_train(self):
        if self._mode == 'augment':
            n = self._n_samples
            one_hot = ops.one_hot(ops.argmax(self._D, axis=1), num_classes=self._n_outputs)
            self._X = ops.repeat(self._X, self._v, axis=0)
            self._D = ops.repeat(self._D, self._v, axis=0)
            one_hot = ops.repeat(one_hot, self._v, axis=0)
            v = ops.reshape(ops.tile([1 / (i + 1) for i in range(self._v)], [n]), (-1, 1))
            self._D += self._D * one_hot * v

    def _make_inputs(self, X):
        temp = ops.reshape(ops.tile([i + 1 for i in range(self._n_outputs)], [X.shape[0]]), (-1, 1))
        if self._mode != 'none':
            temp = ops.one_hot(ops.reshape(temp, (-1, )) - 1, num_classes=self._n_outputs)
        return ops.concatenate([
            ops.cast(ops.repeat(X, self._n_outputs, axis=0), dtype="float32"), ops.cast(temp, dtype="float32")
        ], axis=1)

    def _call(self, X):
        inputs = self._make_inputs(X)
        outputs = self._model(inputs)
        results = ops.reshape(outputs, (X.shape[0], self._n_outputs))
        return keras.activations.softmax(results)


@keras.saving.register_keras_serializable()
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


@keras.saving.register_keras_serializable()
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
