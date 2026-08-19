import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam
from pyldl.algorithms.utils import digamma, inv_digamma, betainc


EPS = np.finfo(np.float32).eps


@keras.saving.register_keras_serializable()
class Delta_LDL(BaseAdam, BaseDeepLDL):
    r""":class:`Delta-LDL <pyldl.algorithms.Delta_LDL>` is proposed in paper :cite:`2025:li`.

    This algorithm is hyperparameter-free.
    """

    @staticmethod
    def _simpson(f, l, r):
        c = (l + r) / 2.
        h = (r - l) / 6.
        return h * (f(l) + 4. * f(c) + f(r))

    @staticmethod
    def _asr(f, l, r, eps, ans, depth):
        mid = (l + r) / 2.
        fl = Delta_LDL._simpson(f, l, mid)
        fr = Delta_LDL._simpson(f, mid, r)
        if abs(fl + fr - ans) <= 15. * eps or depth <= 0:
            return fl + fr + (fl + fr - ans) / 15.
        term1 = Delta_LDL._asr(f, l, mid, eps / 2., fl, depth - 1)
        term2 = Delta_LDL._asr(f, mid, r, eps / 2., fr, depth - 1)
        return term1 + term2

    def _loss(self, X, D, *_):
        D_pred = self._model(X)
        kl = keras.losses.kl_divergence(D, D_pred)

        def f(delta):
            return ops.mean(keras.activations.sigmoid(kl - delta))

        init = Delta_LDL._simpson(f, 0., self._delta)
        return Delta_LDL._asr(f, 0., self._delta, EPS, init, 5)

    def _before_train(self):
        uni = ops.full(ops.shape(self._D), 1. / self._n_outputs)
        self._delta = ops.mean(keras.losses.kl_divergence(self._D, uni))


@keras.saving.register_keras_serializable()
class ApxC_LDL(Delta_LDL):

    @staticmethod
    def _js_divergence(D, D_pred):
        m = (D + D_pred) / 2.
        term1 = keras.losses.kl_divergence(D, m)
        term2 = keras.losses.kl_divergence(D_pred, m)
        return (term1 + term2) / 2.

    @staticmethod
    def _shannon_entropy(x):
        return -ops.sum(x * ops.log(x + EPS), axis=-1)

    @staticmethod
    def jsd_dirichlet_expectation(alpha, beta):
        alpha0 = ops.sum(alpha)
        beta0 = ops.sum(beta)
        gamma = .5 * (alpha / alpha0 + beta / beta0)

        var_alpha = alpha * (alpha0 - alpha) / (alpha0**2 * (alpha0 + 1))
        var_beta  = beta  * (beta0 - beta)  / (beta0**2 * (beta0 + 1))

        term1 = ApxC_LDL._shannon_entropy(gamma)
        term2 = -.125 * ops.sum((var_alpha + var_beta) / gamma)
        term3 = .5 * ops.sum(alpha / alpha0 * (digamma(alpha + 1) - digamma(alpha0 + 1)))
        term4 = .5 * ops.sum(beta / beta0 * (digamma(beta + 1) - digamma(beta0 + 1)))
        return term1 + term2 + term3 + term4

    @staticmethod
    def jsd_dirichlet_variance(alpha, beta):
        alpha0 = ops.sum(alpha)
        beta0 = ops.sum(beta)

        mu = alpha / alpha0
        nu = beta / beta0
        gamma = 0.5 * (mu + nu)

        g_alpha = .5 * ops.log(mu / gamma + 1e-10)
        g_beta = .5 * ops.log(nu / gamma + 1e-10)
        g = ops.concatenate([g_alpha, g_beta], axis=0)

        a_diag = 0.5/mu - 0.25/gamma
        b_diag = -0.25/gamma
        c_diag = 0.5/nu - 0.25/gamma

        denom_a = alpha0**2 * (alpha0 + 1)
        denom_b = beta0**2 * (beta0 + 1)

        def compute_sigma(params, denom):
            params_outer = ops.reshape(params, (-1, 1)) * ops.reshape(params, (1, -1))
            params_sum = ops.sum(params)
            sigma = -params_outer / denom
            sigma = sigma + ops.diag(params * params_sum / denom)
            return sigma
        
        Sigma_d = compute_sigma(alpha, denom_a)
        Sigma_t = compute_sigma(beta, denom_b)

        c = alpha.shape[0]
        zeros_cc = ops.zeros((c, c), dtype=alpha.dtype)

        Sigma = ops.concatenate([
            ops.concatenate([Sigma_d, zeros_cc], axis=1),
            ops.concatenate([zeros_cc, Sigma_t], axis=1)
        ], axis=0)

        H = ops.concatenate([
            ops.concatenate([ops.diag(a_diag), ops.diag(b_diag)], axis=1),
            ops.concatenate([ops.diag(b_diag), ops.diag(c_diag)], axis=1)
        ], axis=0)
        SH = ops.matmul(Sigma, H)

        term1 = ops.sum(g * ops.squeeze(ops.matmul(Sigma, ops.reshape(g, (-1, 1)))))
        term2 = .5 * ops.trace(ops.matmul(SH, SH))
        return term1 + term2

    @staticmethod
    def random_acr(alpha, beta, min_beta=3.):
        mean = ApxC_LDL.jsd_dirichlet_expectation(alpha, beta)
        var = ApxC_LDL.jsd_dirichlet_variance(alpha, beta)

        mean = mean / np.log(2)
        var = var / (np.log(2) * np.log(2))

        var_max = mean * (1 - mean)**2 / (min_beta + 1 - mean)
        var = ops.minimum(var, var_max)

        nu = mean * (1 - mean) / var - 1.0
        alpha_ = mean * nu
        beta_ = (1 - mean) * nu

        def f(delta):
            return betainc(alpha_, beta_, delta / np.log(2))

        init = Delta_LDL._simpson(f, 0., np.log(2))
        res = Delta_LDL._asr(f, 0., np.log(2), EPS, init, 5)
        res /= np.log(2)

        return res

    @staticmethod
    def acr(D, D_pred):
        js = ApxC_LDL._js_divergence(D, D_pred)
        def f(delta):
            return ops.mean(keras.activations.sigmoid(js - delta))
        init = Delta_LDL._simpson(f, 0, np.log(2))
        res = Delta_LDL._asr(f, 0, np.log(2), EPS, init, 5)
        res /= np.log(2)
        return res

    @staticmethod
    def estimate_alpha(D, max_iter=100, tol=1e-7):
        log_p_bar = ops.mean(ops.log(D), axis=0)
        alpha = ops.ones_like(log_p_bar)
        i = ops.zeros((), dtype="int32")

        def cond(i, _):
            return i < max_iter

        def body(i, alpha):
            alpha0 = ops.sum(alpha)
            y = digamma(alpha0) + log_p_bar
            alpha_new = inv_digamma(y)
            diff = ops.max(ops.abs(alpha_new - alpha))
            alpha = ops.clip(ops.where(diff < tol, alpha, alpha_new), EPS, 1e7)
            return i + 1, alpha

        _, alpha = ops.while_loop(
            cond, body, (i, alpha), maximum_iterations=max_iter
        )

        return alpha

    def _loss(self, X, D, *_):
        D_pred = self._model(X)
        beta = ApxC_LDL.estimate_alpha(ops.stop_gradient(D_pred))

        self._l_up = ApxC_LDL.acr(D, D_pred)
        self._l_down = 1 - ApxC_LDL.random_acr(self._alpha_, beta)

        return self._l_up - self._lambda * self._l_down

    def train_step(self, *args, **kwargs):
        super().train_step(*args, **kwargs)
        self._lambda = ops.stop_gradient(self._l_up / (self._l_down + EPS))

    def _before_train(self):
        self._lambda = 1.
        self._n_dirichlet_samples = 100
        self._alpha_ = ApxC_LDL.estimate_alpha(self._D)
