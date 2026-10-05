import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


@keras.saving.register_keras_serializable()
class LDL_LRR(BaseBFGS, BaseDeepLDL):
    """:class:`LDL-LRR <pyldl.algorithms.LDL_LRR>` is proposed in paper :cite:`2023:jia`. 
    LRR refers to *label ranking relation*.

    :term:`BFGS` is used as the optimization algorithm.
    """

    def __init__(self, alpha=1e-3, beta=0., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def ranking_loss(D_pred, P, W, sigma=1e2):
        logsig = lambda x: -ops.logaddexp(0., -x)
        P_hat = sigma * (D_pred[:, :, None] - D_pred[:, None, :])
        l = ((1 - P) * logsig(-P_hat) + P * logsig(P_hat)) * W
        return -ops.sum(l)

    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        D_pred = keras.activations.softmax(self._X @ theta)
        kld = ops.mean(keras.losses.kl_divergence(self._D, D_pred))
        rnk = self.ranking_loss(D_pred, self._P, self._W) / (2 * self._n_samples)
        return kld + self.alpha * rnk + self.beta * self._l2_reg(theta)

    def _before_train(self):
        diff = self._D[:, :, None] - self._D[:, None, :]
        self._P = ops.where(diff > 0., 1., ops.where(diff < 0., 0., .5))
        self._W = ops.square(diff)
