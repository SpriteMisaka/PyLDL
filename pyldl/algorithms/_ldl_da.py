import numpy as np

import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class LDL_DA(BaseAdam, BaseDeepLDL):
    """LDL-DA is proposed in paper Domain Adaptation for Label Distribution Learning.
    """

    @staticmethod
    def augment(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sX = np.concatenate((src, np.zeros(shape=(src.shape[0], tgt.shape[1]))), axis=1)
        tX = np.concatenate((np.zeros(shape=(tgt.shape[0], src.shape[1])), tgt), axis=1)
        return sX, tX

    ORDER_SBU_3DFE = (0, 2, 5, 1, 3, 4)
    @staticmethod
    def reorder_y(y: np.ndarray, order: tuple[int]) -> np.ndarray:
        return y[:, order]

    @staticmethod
    def pairwise_jsd(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        kl = keras.losses.KLDivergence(reduction=tf.compat.v1.losses.Reduction.NONE)
        temp1 = tf.repeat(X, Y.shape[0], axis=0)
        temp2 = tf.tile(Y, [X.shape[0], 1])
        temp3 = 0.5 * (temp1 + temp2)
        js = 0.5 * (kl(temp1, temp3) + kl(temp2, temp3))
        return js

    @staticmethod
    def pairwise_euclidean(X, Y):
        X2 = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), [1, tf.shape(Y)[0]])
        Y2 = tf.tile(tf.reduce_sum(tf.square(Y), axis=1, keepdims=True), [1, tf.shape(X)[0]])
        XY = tf.matmul(X, tf.transpose(Y))
        return X2 + tf.transpose(Y2) - 2 * XY

    @staticmethod
    def pairwise_cosine(X, Y):

        def paired_cosine_distances(X, Y):
            X_norm = tf.nn.l2_normalize(X, axis=1)
            Y_norm = tf.nn.l2_normalize(Y, axis=1)
            return 1 - tf.reduce_sum(tf.multiply(X_norm, Y_norm), axis=1)

        temp1 = tf.repeat(X, Y.shape[0], axis=0)
        temp2 = tf.tile(Y, [X.shape[0], 1])

        cos = tf.abs(1 - paired_cosine_distances(temp1, temp2))
        return tf.reshape(cos, (X.shape[0], Y.shape[0]))

    @staticmethod
    def pairwise_label(X, Y):
        return tf.repeat(X, Y.shape[0]) == tf.tile(Y, [X.shape[0]])

    def __init__(self, n_hidden=256, n_latent=64, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _get_default_model(self):

        encoder = keras.Sequential([keras.layers.InputLayer(input_shape=(self._n_features,)),
                                    keras.layers.Dense(self._n_hidden, activation=keras.layers.LeakyReLU(alpha=0.01)),
                                    keras.layers.Dense(self._n_hidden, activation=keras.layers.LeakyReLU(alpha=0.01)),
                                    keras.layers.Dense(self._n_latent, activation=None)])

        decoder = keras.Sequential([keras.layers.InputLayer(input_shape=(self._n_latent,)),
                                    keras.layers.Dense(self._n_hidden, activation=keras.layers.LeakyReLU(alpha=0.01)),
                                    keras.layers.Dense(self._n_outputs, activation='softmax')])

        return {'encoder': encoder, 'decoder': decoder}

    def _call(self, X, predict=True):
        features = self._model['encoder'](X)
        outputs = self._model['decoder'](features)
        return outputs if predict else (features, outputs)

    @staticmethod
    @tf.function
    def loss_function(y, y_pred):
        return tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))

    def _loss(self, sX, sy, start, end):

        sfeatures, sy_pred = self._call(sX, predict=False)
        tfeatures, ty_pred = self._call(self._tX, predict=False)
        features = tf.concat([sfeatures, tfeatures], axis=0)

        mse = self.loss_function(sy, sy_pred)
        mse += self.loss_function(self._ty, ty_pred)

        dis_X = self.pairwise_euclidean(sfeatures, tfeatures)
        sim_X = tf.maximum(self._margin - dis_X, 0.) if self._margin else self.pairwise_cosine(sfeatures, tfeatures)

        con = tf.reduce_sum(self._hw[start:end] * dis_X) / (tf.reduce_sum(self._hmask_y[start:end]) + EPS)
        con += tf.reduce_sum(self._lw[start:end] * sim_X) / (tf.reduce_sum(self._lmask_y[start:end]) + EPS)

        s_seg = self._seg[start:end]
        t_seg = self._seg[self._sy.shape[0]:]
        seg = tf.concat([s_seg, t_seg], axis=0)

        s_entropy = tf.reshape(self._entropy[start:end], (-1, 1))
        t_entropy = tf.reshape(self._entropy[self._sy.shape[0]:], (-1, 1))
        entropy = tf.concat([s_entropy, t_entropy], axis=0)

        def mwc(W, X, seg, nc):
            return tf.math.unsorted_segment_sum(W * X, seg, nc) / (tf.math.unsorted_segment_sum(W, seg, nc) + EPS)
        p = mwc(entropy, features, seg, self._nc)
        sp = mwc(s_entropy, sfeatures, s_seg, self._nc)
        tp = mwc(t_entropy, tfeatures, t_seg, self._nc)

        mask = tf.reshape(tf.where(
            tf.logical_or(tf.reduce_all(sp == 0., axis=1), tf.reduce_all(tp == 0., axis=1)),
        0., 1.), (-1, 1))
        total = tf.reduce_sum(mask)

        p *= mask
        sp *= mask
        tp *= mask

        pro = tf.reduce_sum(keras.losses.mean_squared_error(sp, tp)) / (total + EPS)
        pro += tf.reduce_sum(keras.losses.mean_squared_error(p, sp)) / (total + EPS)
        pro += tf.reduce_sum(keras.losses.mean_squared_error(p, tp)) / (total + EPS)

        return mse + self._alpha * con + self._beta * pro

    def _before_train(self):

        self._sX = self._X
        self._sy = self._y

        shlabel = tf.argmax(self._sy, axis=1)
        thlabel = tf.argmax(self._ty, axis=1)
        sllabel = tf.argmin(self._sy, axis=1)
        tllabel = tf.argmin(self._ty, axis=1)

        pairs_y = self.pairwise_jsd(self._sy, self._ty)
        pairs_y = tf.cast(
            MinMaxScaler().fit_transform(pairs_y.numpy().reshape(-1, 1)).reshape(-1),
        tf.float32)

        self._hmask_y = tf.where(self.pairwise_label(shlabel, thlabel), 1., 0.)
        self._hw = tf.reshape((1 - pairs_y) * self._hmask_y, (self._sy.shape[0], self._ty.shape[0]))

        self._lmask_y = tf.where(tf.logical_or(self.pairwise_label(sllabel, thlabel),
                                               self.pairwise_label(shlabel, tllabel)), 1., 0.)
        self._lw = tf.reshape(pairs_y * self._lmask_y, (self._sy.shape[0], self._ty.shape[0]))

        self._nc = self._n_outputs * (self._n_outputs - 1)
        concat_y = tf.concat([self._sy, self._ty], axis=0)
        temp = tf.argsort(concat_y, axis=1)[:, -self._r:][:, ::-1]
        _, self._seg = tf.raw_ops.UniqueV2(x=temp, axis=[0])
        self._entropy = -tf.reduce_sum(concat_y * tf.math.log(concat_y + EPS), axis=1)

    def _ft_loss(self, tX, ty, start, end):
        return self.loss_function(ty, self._call(tX))

    def fit(self, sX, sy, tX, ty, ft_epochs=1000, ft_optimizer=None,
            callbacks=None, X_val=None, y_val=None,
            alpha=1e-2, beta=1e-2, r=2, margin=None, fine_tune=True, **kwargs):

        self._tX = tf.cast(tX, tf.float32)
        self._ty = tf.cast(ty, tf.float32)

        self._alpha = alpha
        self._beta = beta
        self._r = r
        self._margin = margin

        super().fit(sX, sy, callbacks=callbacks, X_val=X_val, y_val=y_val,**kwargs)

        if fine_tune:
            self._ft_optimizer = ft_optimizer or self._get_default_optimizer()
            self.train(self._tX, self._ty, ft_epochs, self._tX.shape[0],
                    self._ft_loss, self._model['decoder'].trainable_variables, callbacks, X_val, y_val)

        return self
