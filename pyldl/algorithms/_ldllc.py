import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


@keras.saving.register_keras_serializable()
class LDLLC(BaseBFGS, BaseDeepLDL):
    """:class:`LDLLC <pyldl.algorithms.LDLLC>` is proposed in paper :cite:`2018:jia`.

    :term:`BFGS` is used as optimization algorithm.
    """

    def __init__(self, alpha=1e-3, beta=0., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def _loss(self, params_1d):
        from pyldl.algorithms.utils import pairwise_euclidean, non_diagonal
        theta = self._params2model(params_1d)[0]
        thetaT = ops.transpose(theta)
        D_pred = keras.activations.softmax(self._X @ theta)
        kld = ops.sum(keras.losses.kl_divergence(self._D, D_pred))
        lc = ops.sum(non_diagonal(ops.sign(ops.corrcoef(thetaT)) * pairwise_euclidean(thetaT))) / 2.
        return kld + self.alpha * lc + self.beta * self._l2_reg(theta)
