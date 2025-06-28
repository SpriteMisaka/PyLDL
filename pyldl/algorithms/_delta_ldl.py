import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class Delta_LDL(BaseAdam, BaseDeepLDL):

    def _loss(self, X, D, start, end):
        D_pred = self._model(X)
        kl = keras.losses.kl_divergence(D, D_pred)

        def _f(delta):
            return tf.reduce_mean(keras.activations.sigmoid(kl - delta))

        def _simpson(l, r):
            c = (l + r) / 2.
            h = (r - l) / 6.
            return h * (_f(l) + 4. * _f(c) + _f(r))

        def _asr(l, r, eps, ans, depth):
            mid = (l + r) / 2.
            fl = _simpson(l, mid)
            fr = _simpson(mid, r)
            if abs(fl + fr - ans) <= 15. * eps or depth <= 0:
                return fl + fr + (fl + fr - ans) / 15.
            return _asr(l, mid, eps / 2., fl, depth - 1) + _asr(mid, r, eps / 2., fr, depth - 1)

        return _asr(0, self._delta, EPS, _simpson(0, self._delta), 5)

    def _before_train(self):
        uni = tf.fill(self._D.shape, 1. / self._n_outputs)
        self._delta = tf.reduce_mean(keras.losses.kl_divergence(self._D, uni))
