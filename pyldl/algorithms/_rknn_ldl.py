import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseGD
from pyldl.algorithms.utils import kernel


class RKNN_LDL(BaseGD, BaseDeepLDL):
    r""":class:`RkNN-LDL <pyldl.algorithms.RKNN_LDL>` is proposed in paper :cite:`2025:wang`.
    R\ :math:`k`\NN refers to *residual* :math:`k`\ *-nearest neighbor*.
    """

    def __init__(self, *, k: int = 20, alpha: float = 1e-3,
                 beta: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def _get_default_model(self):
        return self.get_2layer_model(self._K.shape[1] if self._kernel else self._n_features, self._n_outputs)

    def _loss(self, X, D, start, end):
        inputs = self._K if self._kernel else X
        D_pred = self._rho[:, None] * self._D_aaknn + (1 - self._rho)[:, None] * self._call(inputs)
        mse = .5 * ops.sum(keras.losses.mean_squared_error(D, D_pred))
        corr = .5 * ops.sum(keras.losses.mean_squared_error(
            D_pred, (ops.transpose(self._Z) * self._G_dense) @ D_pred
        ))
        reg = .5 * self._l2_reg(self._model)
        return mse + self.alpha * corr + self.beta * reg

    def _before_train(self):
        from ._algorithm_adaptation import AA_KNN
        self._aaknn = AA_KNN(k=self.k)
        X, D = self._to_numpy(self._X), self._to_numpy(self._D)
        self._aaknn.fit(X, D)
        self._D_aaknn = ops.convert_to_tensor(self._aaknn.predict(X), dtype="float32")
        G = self._aaknn._knn.kneighbors_graph().toarray()
        self._G_dense = ops.convert_to_tensor(G, dtype="float32")
        if self._sparse:
            indices = []
            for i in range(self._n_samples):
                columns = np.flatnonzero(G[i])
                j, k = np.meshgrid(columns, columns, indexing='ij')
                indices.append(np.column_stack([
                    np.full(j.size, i), j.reshape(-1), k.reshape(-1)
                ]))
            self._mask_indices = ops.convert_to_tensor(
                np.concatenate(indices), dtype="int32"
            )
        self._rho = self.add_weight(
            name='rho', shape=(self._n_samples,),
            initializer=keras.initializers.RandomUniform(), trainable=True
        )
        self._Z = self.add_weight(
            name='Z', shape=(self._n_samples, self._n_samples),
            initializer=keras.initializers.RandomNormal(), trainable=False
        )
        if self._kernel:
            self._K = ops.convert_to_tensor(kernel(X), dtype="float32")

    def _update_Z(self):
        inputs = self._K if self._kernel else self._X
        D_pred = self._rho[:, None] * self._D_aaknn + (1 - self._rho)[:, None] * self._call(inputs)
        if self._sparse:
            i = self._mask_indices[:, 0]
            j = self._mask_indices[:, 1]
            k = self._mask_indices[:, 2]
            diff = ops.take(D_pred, i, axis=0) - ops.take(D_pred, k, axis=0)
            values = ops.sum(ops.square(diff), axis=1)
            values = ops.where(values != 0., 1. / values, 0.)
            numerator = ops.reshape(
                ops.segment_sum(
                    values, i * self._n_samples + j,
                    self._n_samples * self._n_samples
                ), (self._n_samples, self._n_samples)
            )
            denominator = ops.segment_sum(
                values, i, self._n_samples
            )[:, None]
        else:
            diff = D_pred[:, None, :] - D_pred[None, :, :]
            P = ops.einsum('ild,ijd->ilj', diff, diff)
            mask = (self._G_dense[:, :, None] > 0) & (self._G_dense[:, None, :] > 0)
            C = ops.where(mask, P, 0.)
            C_inv = ops.where(C != 0., 1. / C, 0.)
            numerator = ops.sum(C_inv, axis=2)
            denominator = ops.sum(C_inv, axis=(1, 2))[:, None]
        self._Z.assign(ops.transpose(ops.where(
            self._G_dense != 0, numerator / denominator, 0.
        )))

    def train_step(self, _, *args, **kwargs):
        super().train_step([self._model.trainable_variables, self._optimizer], *args, **kwargs)
        super().train_step([[self._rho], self._rho_optimizer], *args, **kwargs)
        self._rho.assign(ops.clip(self._rho, 0., 1.))
        self._update_Z()

    def predict(self, X):
        X = self._to_numpy(X)
        _, inds = self._aaknn._knn.kneighbors(X)
        rho = np.average(self._to_numpy(self._rho)[inds], axis=1, keepdims=True)
        D_aaknn = np.average(self._aaknn._D[inds], axis=1)
        if self._kernel:
            X = kernel(X, self._to_numpy(self._X))
        return rho * D_aaknn + (1 - rho) * self._to_numpy(self._call(X))

    def fit(self, X, D, *, sparse=True, kernel=True, rho_optimizer=None, **kwargs):
        if kwargs.pop('batch_size', None) is not None:
            raise ValueError("RKNN_LDL does not support 'batch_size'.")
        self._rho_optimizer = rho_optimizer or self._get_default_optimizer()
        self._sparse = sparse
        self._kernel = kernel
        return super().fit(X, D, **kwargs)
