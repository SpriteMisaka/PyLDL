import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


class LDL_DVS(BaseAdam, BaseDeepLDL):
    """:class:`LDL-DVS <pyldl.algorithms.LDL_DVS>` is proposed in paper :cite:`2026:lu`. 
    DVS refers to *divisiveness*.
    """

    def __init__(self, pos: np.ndarray, neg: np.ndarray, *,
                 k: float = 1e1, alpha: float = 1e-1, mode: str = 'ovo', **kwargs):
        """
        :param pos: The positive vector.
        :type pos: np.ndarray
        :param neg: The negative vector.
        :type neg: np.ndarray
        :param k: To control the smoothness of the divisiveness function, default to 10.
        :type k: float
        :param alpha: The weight for the divisiveness loss term.
        :type alpha: float
        :param mode: The mode for computing divisiveness loss, default to 'ovo'.
        :type mode: {None, 'ovo'}
        """
        super().__init__(**kwargs)
        self.pos, self.neg = map(lambda x: ops.cast(x, dtype="float32"), [pos, neg])
        self.k = k
        self.alpha = alpha
        self.mode = mode

    @staticmethod
    def _psi(D, pos, neg, k: float):
        posD, negD = map(lambda x: keras.ops.squeeze(
            ops.matmul(D, keras.ops.expand_dims(x, axis=-1)),
        axis=-1), [pos, neg])
        posexp, negexp = map(lambda x: ops.exp(-k * x), [posD, negD])
        return (posD * posexp + negD * negexp) / (posexp + negexp)

    @staticmethod
    def dvs(psiD, D_pred, pos, neg, k: float):
        charbonnier = lambda x: ops.sqrt(x * x + EPS * EPS)
        psiD_pred = LDL_DVS._psi(D_pred, pos, neg, k)
        return ops.mean(charbonnier(psiD - psiD_pred))

    @staticmethod
    def _ovo_dvs(psiD, D_pred, pos, neg, k: float):
        loss, c = 0., ops.shape(pos)[0]
        for i in range(c):
            _pos = ops.one_hot(i, c, dtype="float32")
            for j in range(c):
                if i == j:
                    continue
                _neg = ops.one_hot(j, c, dtype="float32")
                loss += LDL_DVS.dvs(psiD, D_pred, _pos * pos, _neg * neg, k)
        return loss / float(c * (c - 1))

    def _loss(self, X, D, start, end):
        D_pred = self._model(X)
        kld = ops.mean(keras.losses.kl_divergence(D, D_pred))
        dvs = self._dvs_func(self._psiD[start:end], D_pred, self.pos, self.neg, self.k)
        return kld + self.alpha * dvs

    def _before_train(self):
        self._psiD = LDL_DVS._psi(self._D, self.pos, self.neg, self.k)
        self._dvs_func =LDL_DVS._ovo_dvs if self.mode == 'ovo' else LDL_DVS.dvs
