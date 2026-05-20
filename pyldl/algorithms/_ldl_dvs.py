import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class LDL_DVS(BaseAdam, BaseDeepLDL):

    def __init__(self, pos: np.ndarray, neg: np.ndarray, *,
                 k: float = 1e1, alpha: float = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.pos, self.neg = map(lambda x: tf.cast(x, dtype=tf.float32), [pos, neg])
        self.k = k
        self.alpha = alpha

    @staticmethod
    @tf.function
    def _psi(D: tf.Tensor, pos: tf.Tensor, neg: tf.Tensor, k: float):
        posD, negD = map(lambda x: tf.linalg.matvec(D, x), [pos, neg])
        posexp, negexp = map(lambda x: tf.exp(-k * x), [posD, negD])
        return (posD * posexp + negD * negexp) / (posexp + negexp)

    @staticmethod
    @tf.function
    def dvs(psiD: tf.Tensor, D_pred: tf.Tensor,
            pos: tf.Tensor, neg: tf.Tensor, k: float):
        charbonnier = lambda x: tf.sqrt(x * x + EPS * EPS)
        psiD_pred = LDL_DVS._psi(D_pred, pos, neg, k)
        return tf.reduce_mean(charbonnier(psiD - psiD_pred))

    @staticmethod
    @tf.function
    def _ovo_dvs(psiD: tf.Tensor, D_pred: tf.Tensor,
                 pos: tf.Tensor, neg: tf.Tensor, k: float):
        loss, c = 0., tf.shape(pos)[0]
        for i in range(c):
            _pos = tf.one_hot(i, c, dtype=tf.float32)
            for j in range(c):
                if i == j:
                    continue
                _neg = tf.one_hot(j, c, dtype=tf.float32)
                loss += LDL_DVS.dvs(psiD, D_pred, _pos * pos, _neg * neg, k)
        return loss / float(c * (c - 1))

    @tf.function
    def _loss(self, X, D, start, end):
        D_pred = self._model(X)
        kld = tf.reduce_mean(keras.losses.kl_divergence(D, D_pred))
        dvs = LDL_DVS._ovo_dvs(self._psiD[start:end], D_pred, self.pos, self.neg, self.k)
        return kld + self.alpha * dvs

    def _before_train(self):
        self._psiD = LDL_DVS._psi(self._D, self.pos, self.neg, self.k)
