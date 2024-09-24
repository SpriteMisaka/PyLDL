from typing import Optional
import numpy as np

import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from pyldl.algorithms.utils import pairwise_euclidean
from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class LDL_DA(BaseAdam, BaseDeepLDL):
    """:class:`LDL-DA <pyldl.algorithms.LDL_DA>` is proposed in paper :cite:`2024:wu`.
    """

    @staticmethod
    def augment(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feature augmentation.

        :param src: Source data (shape: :math:`[m_s,\, n_s]`).
        :type src: np.ndarray
        :param tgt: Target data (shape: :math:`[m_t,\, n_t]`).
        :type tgt: np.ndarray
        :return: Augmented source and target data (shape: :math:`[m_s,\, n_s + n_t]` and :math:`[m_t,\, n_s + n_t]`, respectively).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        sX = np.concatenate((src, np.zeros(shape=(src.shape[0], tgt.shape[1]))), axis=1)
        tX = np.concatenate((np.zeros(shape=(tgt.shape[0], src.shape[1])), tgt), axis=1)
        return sX, tX

    ORDER_SBU_3DFE = (0, 2, 5, 1, 3, 4)
    @staticmethod
    def reorder_y(y: np.ndarray, order: tuple[int]) -> np.ndarray:
        """Reorder label distributions for consistent label semantics.

        :param y: Label distributions.
        :type y: np.ndarray
        :param order: New order.
        :type order: tuple[int]
        :return: Reordered label distributions.
        :rtype: np.ndarray
        """
        return y[:, order]

    @staticmethod
    def pairwise_jsd(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """Pairwise Jensen-Shannon divergence.

        :param X: Matrix :math:`\\boldsymbol{X}` (shape: :math:`[m_X,\, n_X]`).
        :type X: tf.Tensor
        :param Y: Matrix :math:`\\boldsymbol{Y}` (shape: :math:`[m_Y,\, n_Y]`).
        :type Y: tf.Tensor
        :return: Pairwise Jensen-Shannon divergence (shape: :math:`[m_X,\, m_Y]`).
        :rtype: tf.Tensor
        """
        kl = keras.losses.KLDivergence(reduction=tf.compat.v1.losses.Reduction.NONE)
        temp1 = tf.repeat(X, Y.shape[0], axis=0)
        temp2 = tf.tile(Y, [X.shape[0], 1])
        temp3 = 0.5 * (temp1 + temp2)
        js = 0.5 * (kl(temp1, temp3) + kl(temp2, temp3))
        return js

    @staticmethod
    def pairwise_cosine(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """Pairwise cosine similarity.

        :param X: Matrix :math:`\\boldsymbol{X}` (shape: :math:`[m_X,\, n_X]`).
        :type X: tf.Tensor
        :param Y: Matrix :math:`\\boldsymbol{Y}` (shape: :math:`[m_Y,\, n_Y]`).
        :type Y: tf.Tensor
        :return: Pairwise cosine similarity (shape: :math:`[m_X,\, m_Y]`).
        :rtype: tf.Tensor
        """

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
        """Pairwise label comparison. True if two labels are the same, otherwise False.

        :param X: Matrix :math:`\\boldsymbol{X}` (shape: :math:`[m_X,\, n_X]`).
        :type X: tf.Tensor
        :param Y: Matrix :math:`\\boldsymbol{Y}` (shape: :math:`[m_Y,\, n_Y]`).
        :type Y: tf.Tensor
        :return: Pairwise label comparison (shape: :math:`[m_X,\, m_Y]`).
        :rtype: tf.Tensor
        """
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

        dis_X = pairwise_euclidean(sfeatures, tfeatures)
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

    def fit(self, sX: np.ndarray, sy: np.ndarray, tX: np.ndarray, ty: np.ndarray, *, callbacks=None, X_val=None, y_val=None,
            ft_epochs : int = 1000, ft_optimizer : Optional[keras.optimizers.Optimizer] = None,
            alpha : float = 1e-2, beta : float = 1e-2, r : int = 2, margin : Optional[float] = None, fine_tune : bool = True, **kwargs):
        """Fit the model.

        :param sX: Source features.
        :type sX: np.ndarray
        :param sy: Source label distributions.
        :type sy: np.ndarray
        :param tX: Target features.
        :type tX: np.ndarray
        :param ty: Target label distributions.
        :type ty: np.ndarray
        :param ft_epochs: Fine-tuning epochs, defaults to 1000.
        :type ft_epochs: int, optional
        :param ft_optimizer: Fine-tuning optimizer, if None, the default optimizer is used, defaults to None.
        :type ft_optimizer: keras.optimizers.Optimizer, optional
        :param alpha: Hyperparameter to control the contrastive alignment loss, defaults to 1e-2.
        :type alpha: float
        :param beta: Hyperparameter to control the prototype alignment loss, defaults to 1e-2.
        :type beta: float
        :param r: Number of prototypes, defaults to 2.
        :type r: int
        :param margin: Margin for the similarity measure, defaults to None. If None, cosine similarity is used; otherwise, max-margin euclidean distance is used.
        :type margin: float, optional
        :param fine_tune: Whether to fine-tune the model, defaults to True.
        :type fine_tune: bool, optional
        :return: Fitted model.
        :rtype: LDL_DA
        """

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
