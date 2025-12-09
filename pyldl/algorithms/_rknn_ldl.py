import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseGD
from pyldl.algorithms.utils import csr2sparse, kernel


class RKNN_LDL(BaseGD, BaseDeepLDL):
    r""":class:`RkNN-LDL <pyldl.algorithms.RKNN_LDL>` is proposed in paper :cite:`2025:wang`. 
    R\ :math:`k`\NN refers to *residual* :math:`k`\ *-nearest neighbor*.
    """

    def __init__(self, *, k: int = 20, alpha: float = 1e-3, beta: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def _get_default_model(self):
        return self.get_2layer_model(self._K.shape[1] if self._kernel else self._n_features, self._n_outputs)

    @tf.function
    def _loss(self, X, D, start, end):
        inputs = self._K if self._kernel else X
        D_pred = self._rho[:, None] * self._D_aaknn + (1 - self._rho)[:, None] * self._call(inputs)
        mse = .5 * tf.reduce_sum(keras.losses.mean_squared_error(D, D_pred))
        corr = .5 * tf.reduce_sum(keras.losses.mean_squared_error(D_pred, (tf.transpose(self._Z) * self._G_dense) @ D_pred))
        reg = .5 * self._l2_reg(self._model)
        return mse + self.alpha * corr + self.beta * reg

    def _before_train(self):
        from ._algorithm_adaptation import AA_KNN
        self._aaknn = AA_KNN(k=self.k)
        X, D = self._X.numpy(), self._D.numpy()
        self._aaknn.fit(X, D)
        self._D_aaknn = tf.convert_to_tensor(self._aaknn.predict(X), dtype=tf.float32)
        self._G_dense = tf.convert_to_tensor(self._aaknn._knn.kneighbors_graph().toarray(), dtype=tf.float32)
        if self._sparse:
            self._G_sparse = csr2sparse(self._aaknn._knn.kneighbors_graph())
            self._mask_indices, self._mask_shape = self.construct_mask(self._G_sparse)
        self._rho = tf.Variable(tf.random.uniform((self._n_samples,)), trainable=True)
        self._Z = tf.Variable(tf.random.normal((self._n_samples, self._n_samples)), trainable=True)
        if self._kernel:
            self._K = kernel(self._X)

    @staticmethod
    def construct_mask(G):
        indices = G.indices
        dense_shape = G.dense_shape
        rows = indices[:, 0]
        cols = indices[:, 1]

        mask_indices = []
        unique_rows = tf.unique(rows).y
        for i in unique_rows:
            row_mask = (rows == i)
            cols_in_row = cols[row_mask]
            j, k = tf.meshgrid(cols_in_row, cols_in_row)
            j_k_pairs = tf.stack([j, k], axis=-1)
            j_k_pairs = tf.reshape(j_k_pairs, [-1, 2])

            i_indices = tf.fill([j_k_pairs.shape[0]], i)
            i_j_k_indices = tf.stack([i_indices, j_k_pairs[:, 0], j_k_pairs[:, 1]], axis=1)
            mask_indices.append(i_j_k_indices)

        mask_indices = tf.concat(mask_indices, axis=0)
        mask_shape = (dense_shape[0], dense_shape[1], dense_shape[1])
        return mask_indices, mask_shape

    @staticmethod
    @tf.function
    def construct_C(mask_indices, mask_shape, D_pred):
        i = mask_indices[:, 0]
        j = mask_indices[:, 2]

        D_pred_i = tf.gather(D_pred, i)
        D_pred_j = tf.gather(D_pred, j)
        diff = D_pred_i - D_pred_j

        C_values = tf.reduce_sum(diff * diff, axis=1)
        return tf.sparse.reorder(
            tf.sparse.SparseTensor(mask_indices, C_values, mask_shape)
        )

    @tf.function
    def _update_Z(self):
        inputs = self._K if self._kernel else self._X
        D_pred = self._rho[:, None] * self._D_aaknn + (1 - self._rho)[:, None] * self._call(inputs)

        if self._sparse:
            C = self.construct_C(self._mask_indices, self._mask_shape, D_pred)
            C_inv = tf.sparse.map_values(tf.pow, C, -1)
            reduce_sum = tf.sparse.reduce_sum
        else:
            diff = D_pred[:, None, :] - D_pred[None, :, :]
            P = tf.einsum('ild,ijd->ilj', diff, diff)
            mask = (self._G_dense[:, :, None] > 0) & (self._G_dense[:, None, :] > 0)
            C = tf.where(mask, P, 0.)
            C_inv = tf.where(C != 0., 1. / C, 0.)
            reduce_sum = tf.reduce_sum

        numerator = reduce_sum(C_inv, axis=2)
        denominator = reduce_sum(C_inv, axis=[1, 2])[:, None]
        self._Z.assign(tf.transpose(tf.where(self._G_dense != 0, numerator / denominator, 0.)))

    def train_step(self, batch, loss, _, epoch, epochs, start, end):
        super().train_step(batch, loss, self._model.trainable_variables, epoch, epochs, start, end)
        super().train_step(batch, loss, [self._rho], epoch, epochs, start, end)
        self._rho.assign(tf.clip_by_value(self._rho, 0., 1.))
        self._update_Z()

    def predict(self, X):
        _, inds = self._aaknn._knn.kneighbors(X)
        rho = np.average(self._rho.numpy()[inds], axis=1, keepdims=True)
        D_aaknn = np.average(self._aaknn._D[inds], axis=1)
        if self._kernel:
            X = kernel(X, self._X)
        return rho * D_aaknn + (1 - rho) * self._call(X).numpy()

    def fit(self, X, D, *, sparse=True, kernel=True, **kwargs):
        if kwargs.pop('batch_size', None) is not None:
            raise ValueError("RKNN_LDL does not support 'batch_size'.")
        self._sparse = sparse
        self._kernel = kernel
        return super().fit(X, D, **kwargs)
