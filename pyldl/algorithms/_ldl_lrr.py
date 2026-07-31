import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


EPS = np.finfo(np.float32).eps


@keras.saving.register_keras_serializable()
class LDL_LRR(BaseBFGS, BaseDeepLDL):
    """:class:`LDL-LRR <pyldl.algorithms.LDL_LRR>` is proposed in paper :cite:`2023:jia`. 
    LRR refers to *label ranking relation*.

    :term:`BFGS` is used as the optimization algorithm.
    """

    def __init__(self, alpha=1e-2, beta=0., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def ranking_loss(D_pred, P, W):
        P_hat = ops.sigmoid(D_pred[:, :, None] - D_pred[:, None, :])
        l = ((1 - P) * ops.log(ops.clip(1 - P_hat, EPS, 1.)) +
              P * ops.log(ops.clip(P_hat, EPS, 1.))) * W
        return -ops.sum(l)

    @staticmethod
    def preprocessing(D):
        diff = D[:, :, None] - D[:, None, :]
        return ops.where(diff > .5, 1., 0.), ops.square(diff)

    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        D_pred = keras.activations.softmax(self._X @ theta)
        kld = ops.mean(keras.losses.kl_divergence(self._D, D_pred))
        rnk = self.ranking_loss(D_pred, self._P, self._W) / (2 * self._n_samples)
        return kld + self.alpha * rnk + self.beta * self._l2_reg(theta)

    def _before_train(self):
        self._P, self._W = LDL_LRR.preprocessing(self._D)
