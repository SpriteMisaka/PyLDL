import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseBFGS


@keras.saving.register_keras_serializable()
class LDL_DPA(BaseBFGS, BaseDeepLDL):
    """:class:`LDL-DPA <pyldl.algorithms.LDL_DPA>` is proposed in paper :cite:`2024:jia`. 
    DPA refers to *description-degree percentile average*.

    :term:`BFGS` is used as the optimization algorithm.
    """

    def __init__(self, alpha=1e-3, beta=1e-3, gamma=0., **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    @staticmethod
    def rnkdpa(R, D_pred):
        return -ops.sum(ops.mean(R * D_pred, axis=1))

    @staticmethod
    def disvar(D, D_pred):
        return ops.sum((ops.var(D, axis=1) - ops.var(D_pred, axis=1))**2)

    def _loss(self, params_1d):
        theta = self._params2model(params_1d)[0]
        D_pred = keras.activations.softmax(self._X @ theta)
        kld = ops.sum(keras.losses.kl_divergence(self._D, D_pred))
        rnkdpa = self.rnkdpa(self._R, D_pred)
        disvar = self.disvar(self._D, D_pred)
        return kld + self._alpha * rnkdpa + self._beta * disvar + self._gamma * self._l2_reg(theta)

    def _before_train(self):
        from scipy.stats import rankdata
        self._R = ops.cast(rankdata(self._D, axis=1), dtype="float32")
