import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


@keras.saving.register_keras_serializable()
class Duo_LDL(BaseAdam, BaseDeepLDL):
    """:class:`Duo-LDL <pyldl.algorithms.Duo_LDL>` is proposed in paper :cite:`2021:zychowski`.
    """

    def __init__(self, n_hidden=40, **kwargs):
        super().__init__(n_hidden, **kwargs)

    def _get_default_model(self):
        output_units = self._n_outputs * (self._n_outputs - 1)
        return self.get_3layer_model(self._n_features, self._n_hidden, output_units, output_activation='tanh')

    def _loss(self, X, _, start, end):
        return self.loss_function(self._C[start:end], self._call(X))

    def _before_train(self):
        self._C = ops.concatenate(
            [self._D - ops.roll(self._D, i, axis=1) for i in range(1, self._n_outputs)]
        , axis=1)

    def predict(self, X):
        from pyldl.algorithms.utils import normalize
        C_pred = self._call(X)
        shape = (X.shape[0], self._n_outputs - 1, self._n_outputs)
        C_pred_reshaped = ops.transpose(ops.reshape(C_pred, shape), (0, 2, 1))
        D_pred = self._to_numpy((ops.sum(C_pred_reshaped, axis=2) + 1) / self._n_outputs)
        return normalize(D_pred)
