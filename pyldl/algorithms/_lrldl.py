import numpy as np

from pyldl.algorithms.utils import svt, binaryzation
from pyldl.algorithms.base import BaseADMM, BaseLDL


class _LRLDL(BaseADMM, BaseLDL):
    r"""Base class for :class:`pyldl.algorithms.TLRLDL` and :class:`pyldl.algorithms.TKLRLDL`, 
    which are proposed in paper :cite:`2024:kou`. LR refers to *low-rank*.

    :term:`ADMM` is used as optimization algorithm.
    """

    def __init__(self, mode='threshold', param=None, alpha=1e-3, beta=1e-3, random_state=None):
        super().__init__(random_state)
        self._mode = mode
        self._param = param
        self._alpha = alpha
        self._beta = beta

    def _update_W(self):
        r"""Please note that Eq. (8) in paper :cite:`2024:kou` should be corrected to:

        .. math::

            \begin{aligned}
            \boldsymbol{W} \leftarrow & \left(\left(\mu \boldsymbol{G} + \boldsymbol{\Gamma}_1 + \boldsymbol{L}\right)
            \boldsymbol{O}^{\top} \boldsymbol{X} + \boldsymbol{D}\boldsymbol{X} \right) \\
            & \left( \boldsymbol{X}^{\top}\boldsymbol{X} + 2 \lambda \boldsymbol{I} +
            (1+\mu) \boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{O}^{\top}\boldsymbol{X} \right)^{-1}\text{,}
            \end{aligned}

        where :math:`\\boldsymbol{I}` is the identity matrix.

        And Eq. (10) should be corrected to:

        .. math::

            \begin{aligned}
            \boldsymbol{O} \leftarrow & \left( (1+\mu) \boldsymbol{X}\boldsymbol{W}^{\top}
            \left( \boldsymbol{X}\boldsymbol{W}^{\top} \right)^{\top} + 2 \lambda \boldsymbol{I} \right)^{-1} \\
            & \boldsymbol{X}\boldsymbol{W}^{\top} \left(\boldsymbol{L} + \mu \boldsymbol{G} - \boldsymbol{\Gamma}_1\right)\text{.}
            \end{aligned}
        """
        OTX = self._O.T @ self._X
        temp1 = (self._rho * self._Z.T - self._V.T + self._L.T) @ OTX + self._D.T @ self._X
        temp2 = self._X.T @ self._X + 2 * self._I1 + (1 + self._rho) * self._X.T @ self._O @ OTX
        self._W = np.transpose(temp1 @ np.linalg.inv(temp2))

        WXT = self._W.T @ self._X.T
        temp1 = (1 + self._rho) * WXT.T @ WXT + 2 * self._I2
        temp2 = WXT.T @ (self._L.T + self._rho * self._Z.T - self._V.T)
        self._O = np.linalg.inv(temp1) @ temp2

    def _update_Z(self):
        A = self._W.T @ self._X.T @ self._O + self._V.T / self._rho
        tau = self._alpha / self._rho
        self._Z = np.transpose(svt(A, tau))

    def _update_V(self):
        r"""Please note that Eq. (11) in paper :cite:`2024:kou` should be corrected to:

        .. math::

            \boldsymbol{\Gamma}_1 \leftarrow \boldsymbol{\Gamma}_1 + \mu
            \left(\boldsymbol{W}\boldsymbol{X}^{\top}\boldsymbol{O} - \boldsymbol{G}\right)\text{.}
        """
        self._V = np.transpose(self._V.T + self._rho * (self._W.T @ self._X.T @ self._O - self._Z.T))
        self._rho *= 1.1

    def _before_train(self):
        self._O = np.random.random((self._n_samples, self._n_samples))
        self._L = binaryzation(self._D, method=self._mode, param=self._param)

        self._I1 = self._beta * np.eye(self._n_features)
        self._I2 = self._beta * np.eye(self._n_samples)

    @property
    def constraint(self):
        return [[self._Z.T, self._W.T @ self._X.T @ self._O]]

    @property
    def params(self):
        return [self._W.T, self._O, self._Z.T]

    def fit(self, X, D, rho=1e-3, stopping_criterion='error', **kwargs):
        return super().fit(X, D, rho=rho, stopping_criterion=stopping_criterion, **kwargs)


class TLRLDL(_LRLDL):
    r""":class:`TLRLDL <pyldl.algorithms.TLRLDL>` is proposed in paper :cite:`2024:kou`. 
    T refers to *threshold* (a threshold-based :class:`binaryzation <pyldl.algorithms.utils.binaryzation>` method is used to generate the logical label matrix).
    """

    def __init__(self, **kwargs):
        super().__init__('threshold', **kwargs)


class TKLRLDL(_LRLDL):
    r""":class:`TKLRLDL <pyldl.algorithms.TKLRLDL>` is proposed in paper :cite:`2024:kou`. 
    TK refers to *top-*\ :math:`k` (a top-:math:`k` :class:`binaryzation <pyldl.algorithms.utils.binaryzation>` method is used to generate the logical label matrix).
    """

    def __init__(self, **kwargs):
        super().__init__('topk', **kwargs)
