import numpy as np

from pyldl.algorithms.utils import svt, binaryzation
from pyldl.algorithms.base import BaseADMM, BaseLDL


class _LRLDL(BaseADMM, BaseLDL):
    """Base class for :class:`pyldl.algorithms.TLRLDL` and :class:`pyldl.algorithms.TKLRLDL`.

    :term:`ADMM` is used as optimization algorithm.
    """

    def __init__(self, mode='threshold', param=None, random_state=None):
        super().__init__(random_state)
        self._mode = mode
        self._param = param

    def _update_W(self):
        """Please note that Eq. (8) in paper :cite:`2024:kou` should be corrected to:

        .. math::

            \\begin{aligned}
            \\boldsymbol{W} \\leftarrow & \\left(\\left(\\mu \\boldsymbol{G} + \\boldsymbol{\\Gamma}_1 + \\boldsymbol{L}\\right)
            \\boldsymbol{O}^{\\text{T}} \\boldsymbol{X} + \\boldsymbol{D}\\boldsymbol{X} \\right) \\\\
            & \\left( \\boldsymbol{X}^{\\text{T}}\\boldsymbol{X} + 2 \\lambda \\boldsymbol{I} +
            (1+\\mu) \\boldsymbol{X}^{\\text{T}}\\boldsymbol{O}\\boldsymbol{O}^{\\text{T}}\\boldsymbol{X} \\right)^{-1}\\text{,}
            \\end{aligned}

        where :math:`\\boldsymbol{I}` is the identity matrix.

        And Eq. (10) should be corrected to:

        .. math::

            \\begin{aligned}
            \\boldsymbol{O} \\leftarrow & \\left( (1+\\mu) \\boldsymbol{X}\\boldsymbol{W}^{\\text{T}}
            \\left( \\boldsymbol{X}\\boldsymbol{W}^{\\text{T}} \\right)^{\\text{T}} + 2 \\lambda \\boldsymbol{I} \\right)^{-1} \\\\
            & \\boldsymbol{X}\\boldsymbol{W}^{\\text{T}} \\left(\\boldsymbol{L} + \\mu \\boldsymbol{G} - \\boldsymbol{\\Gamma}_1\\right)\\text{.}
            \\end{aligned}
        """
        OTX = self._O.T @ self._X
        temp1 = (self._rho * self._Z.T - self._V.T + self._L.T) @ OTX + self._y.T @ self._X
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
        """Please note that Eq. (11) in paper :cite:`2024:kou` should be corrected to:

        .. math::

            \\boldsymbol{\\Gamma}_1 \\leftarrow \\boldsymbol{\\Gamma}_1 + \\mu
            \\left(\\boldsymbol{W}\\boldsymbol{X}^{\\text{T}}\\boldsymbol{O} - \\boldsymbol{G}\\right)\\text{.}
        """
        self._V = np.transpose(self._V.T + self._rho * (self._W.T @ self._X.T @ self._O - self._Z.T))
        self._rho *= 1.1

    def _before_train(self):
        self._O = np.random.random((self._X.shape[0], self._X.shape[0]))
        self._L = binaryzation(self._y, method=self._mode, param=self._param)

        self._I1 = self._beta * np.eye(self._n_features)
        self._I2 = self._beta * np.eye(self._X.shape[0])

    @property
    def constraint(self):
        return [[self._Z.T, self._W.T @ self._X.T @ self._O]]

    @property
    def params(self):
        return [self._W.T, self._O, self._Z.T]

    def fit(self, X, y, alpha=1e-3, beta=1e-3,
            rho=1e-3, stopping_criterion='error', **kwargs):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, y, rho=rho, stopping_criterion=stopping_criterion, **kwargs)


class TLRLDL(_LRLDL):
    """:class:`TLRLDL <pyldl.algorithms.TLRLDL>` is proposed in paper :cite:`2024:kou`.

    A threshold-based :class:`binaryzation <pyldl.algorithms.utils.binaryzation>` method is used to generate the logical label matrix.
    """

    def __init__(self, param=None, random_state=None):
        super().__init__('threshold', param, random_state)


class TKLRLDL(_LRLDL):
    """:class:`TKLRLDL <pyldl.algorithms.TKLRLDL>` is proposed in paper :cite:`2024:kou`.

    A top-:math:`k` :class:`binaryzation <pyldl.algorithms.utils.binaryzation>` method is used to generate the logical label matrix.
    """

    def __init__(self, param=None, random_state=None):
        super().__init__('topk', param, random_state)
