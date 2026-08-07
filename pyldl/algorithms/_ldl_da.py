from typing import Optional
import numpy as np

import keras
import keras.ops as ops
from sklearn.preprocessing import MinMaxScaler

from pyldl.algorithms.utils import pairwise_euclidean, pairwise_cosine
from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class LDL_DA(BaseAdam, BaseDeepLDL):
    """:class:`LDL-DA <pyldl.algorithms.LDL_DA>` is proposed in paper :cite:`2025:wu`.
    """

    @staticmethod
    def augment(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sX = np.concatenate((src, np.zeros(shape=(src.shape[0], tgt.shape[1]))), axis=1)
        tX = np.concatenate((np.zeros(shape=(tgt.shape[0], src.shape[1])), tgt), axis=1)
        return sX, tX

    ORDER_SBU_3DFE = (0, 2, 5, 1, 3, 4)
    @staticmethod
    def reorder_D(D: np.ndarray, order: tuple[int]) -> np.ndarray:
        return D[:, order]

    @staticmethod
    def pairwise_jsd(X, Y):
        temp1 = X[:, None, :]
        temp2 = Y[None, :, :]
        temp3 = 0.5 * (temp1 + temp2)
        return 0.5 * (
            keras.losses.kl_divergence(temp1, temp3)
            + keras.losses.kl_divergence(temp2, temp3)
        )

    @staticmethod
    def pairwise_label(X, Y):
        return X[:, None] == Y[None, :]

    def __init__(self, n_hidden=256, n_latent=64,
                 alpha : float = 1e-2, beta : float = 1e-2, r : int = 2, margin : Optional[float] = None, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._r = r
        self._margin = margin

    def _get_default_model(self):
        encoder = self.get_3layer_model(self._n_features, self._n_hidden, self._n_latent,
                                        hidden_activation=keras.layers.LeakyReLU(alpha=0.01), output_activation=None)
        decoder = self.get_3layer_model(self._n_latent, self._n_hidden, self._n_outputs,
                                        hidden_activation=keras.layers.LeakyReLU(alpha=0.01), output_activation='softmax')
        return {'encoder': encoder, 'decoder': decoder}

    def _call(self, X, predict=True):
        features = self._model['encoder'](X)
        outputs = self._model['decoder'](features)
        return outputs if predict else (features, outputs)

    @staticmethod
    def loss_function(D, D_pred):
        return ops.mean(keras.losses.kl_divergence(D, D_pred))

    def _loss(self, sX, sD, start, end):

        sfeatures, sD_pred = self._call(sX, predict=False)
        tfeatures, tD_pred = self._call(self._tX, predict=False)
        features = ops.concatenate([sfeatures, tfeatures], axis=0)

        mse = self.loss_function(sD, sD_pred)
        mse += self.loss_function(self._tD, tD_pred)

        dis_X = pairwise_euclidean(sfeatures, tfeatures)
        sim_X = ops.maximum(self._margin - dis_X, 0.) if self._margin else pairwise_cosine(sfeatures, tfeatures)

        con = ops.sum(self._hw[start:end] * dis_X) / (ops.sum(self._hmask_D[start:end]) + EPS)
        con += ops.sum(self._lw[start:end] * sim_X) / (ops.sum(self._lmask_D[start:end]) + EPS)

        s_seg = self._seg[start:end]
        t_seg = self._seg[self._sD.shape[0]:]
        seg = ops.concatenate([s_seg, t_seg], axis=0)

        s_entropy = ops.reshape(self._entropy[start:end], (-1, 1))
        t_entropy = ops.reshape(self._entropy[self._sD.shape[0]:], (-1, 1))
        entropy = ops.concatenate([s_entropy, t_entropy], axis=0)

        def mwc(W, X, seg, nc):
            return ops.segment_sum(W * X, seg, nc) / (ops.segment_sum(W, seg, nc) + EPS)
        p = mwc(entropy, features, seg, self._nc)
        sp = mwc(s_entropy, sfeatures, s_seg, self._nc)
        tp = mwc(t_entropy, tfeatures, t_seg, self._nc)

        mask = ops.reshape(ops.where(
            ops.logical_or(ops.all(sp == 0., axis=1), ops.all(tp == 0., axis=1)),
        0., 1.), (-1, 1))
        total = ops.sum(mask)

        p *= mask
        sp *= mask
        tp *= mask

        pro = ops.sum(keras.losses.mean_squared_error(sp, tp)) / (total + EPS)
        pro += ops.sum(keras.losses.mean_squared_error(p, sp)) / (total + EPS)
        pro += ops.sum(keras.losses.mean_squared_error(p, tp)) / (total + EPS)

        return mse + self._alpha * con + self._beta * pro

    def _before_train(self):

        self._sX = self._X
        self._sD = self._D

        shlabel = ops.argmax(self._sD, axis=1)
        thlabel = ops.argmax(self._tD, axis=1)
        sllabel = ops.argmin(self._sD, axis=1)
        tllabel = ops.argmin(self._tD, axis=1)

        pairs_y = self.pairwise_jsd(self._sD, self._tD)
        pairs_shape = pairs_y.shape
        pairs_y = ops.convert_to_tensor(
            MinMaxScaler().fit_transform(
                self._to_numpy(pairs_y).reshape(-1, 1)
            ).reshape(pairs_shape), dtype="float32"
        )

        self._hmask_D = ops.where(self.pairwise_label(shlabel, thlabel), 1., 0.)
        self._hw = ops.reshape((1 - pairs_y) * self._hmask_D, (self._sD.shape[0], self._tD.shape[0]))

        self._lmask_D = ops.where(ops.logical_or(self.pairwise_label(sllabel, thlabel),
                                                self.pairwise_label(shlabel, tllabel)), 1., 0.)
        self._lw = ops.reshape(pairs_y * self._lmask_D, (self._sD.shape[0], self._tD.shape[0]))

        self._nc = self._n_outputs * (self._n_outputs - 1)
        concat_D = ops.concatenate([self._sD, self._tD], axis=0)
        _, temp = ops.top_k(concat_D, k=self._r)
        _, seg = np.unique(self._to_numpy(temp), axis=0, return_inverse=True)
        self._seg = ops.convert_to_tensor(seg, dtype="int32")
        self._entropy = -ops.sum(concat_D * ops.log(concat_D + EPS), axis=1)

    def _ft_loss(self, tX, ty, start, end):
        return self.loss_function(ty, self._call(tX))

    def _prepare_validation_data(self, X, Y, validation_split):
        (self._tX, self._tD), (self._tX_val, self._tD_val) = self._split_validation_data(
            validation_split, self._tX, self._tD
        )
        self._tX = ops.cast(self._tX, "float32")
        self._tD = ops.cast(self._tD, "float32")
        return X, Y, self._tX_val, self._tD_val, None

    def fit(self, sX: np.ndarray, sD: np.ndarray, tX: np.ndarray, tD: np.ndarray, *, callbacks=None,
            validation_split=0., ft_epochs : int = 1000,
            ft_optimizer : Optional[keras.optimizers.Optimizer] = None,
            fine_tune : bool = True, **kwargs):

        self._tX = tX
        self._tD = tD
        super().fit(sX, sD, callbacks=callbacks,
                    validation_split=validation_split, **kwargs)

        if fine_tune:
            self._ft_optimizer = ft_optimizer or self._get_default_optimizer()
            original_loss = self._loss
            self._loss = self._ft_loss
            self._optimizer = self._ft_optimizer
            self._model['encoder'].trainable = False
            self.train(self._tX, self._tD, ft_epochs, callbacks,
                       self._tX_val, self._tD_val, None)
            self._model['encoder'].trainable = True
            self._loss = original_loss

        return self
