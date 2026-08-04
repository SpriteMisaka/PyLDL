import numpy as np

import keras
import keras.ops as ops
import tensorflow as tf

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps

EULER_GAMMA = np.euler_gamma


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
    @tf.function
    def _js_divergence(D, D_pred):
        m = (D + D_pred) / 2.
        term1 = keras.losses.kl_divergence(D, m)
        term2 = keras.losses.kl_divergence(D_pred, m)
        return (term1 + term2) / 2.

    @staticmethod
    @tf.function
    def _shannon_entropy(x):
        return -tf.reduce_sum(x * tf.math.log(x + EPS), axis=-1)

    @staticmethod
    @tf.function
    def jsd_dirichlet_expectation(alpha, beta):
        alpha0 = tf.reduce_sum(alpha)
        beta0  = tf.reduce_sum(beta)
        gamma = .5 * (alpha / alpha0 + beta / beta0)

        var_alpha = alpha * (alpha0 - alpha) / (alpha0**2 * (alpha0 + 1))
        var_beta  = beta  * (beta0 - beta)  / (beta0**2 * (beta0 + 1))

        term1 = ApxC_LDL._shannon_entropy(gamma)
        term2 = -.125 * tf.reduce_sum((var_alpha + var_beta) / gamma)
        term3 = .5 * tf.reduce_sum(alpha / alpha0 * (tf.math.digamma(alpha + 1) - tf.math.digamma(alpha0 + 1)))
        term4 = .5 * tf.reduce_sum(beta / beta0  * (tf.math.digamma(beta + 1)  - tf.math.digamma(beta0 + 1)))
        return term1 + term2 + term3 + term4

    @staticmethod
    @tf.function
    def jsd_dirichlet_variance(alpha, beta):
        alpha0 = tf.reduce_sum(alpha)
        beta0 = tf.reduce_sum(beta)

        mu = alpha / alpha0
        nu = beta / beta0
        gamma = 0.5 * (mu + nu)

        g_alpha = 0.5 * tf.math.log(mu / gamma + 1e-10)
        g_beta = 0.5 * tf.math.log(nu / gamma + 1e-10)
        g = tf.concat([g_alpha, g_beta], axis=0)

        a_diag = 0.5/mu - 0.25/gamma
        b_diag = -0.25/gamma
        c_diag = 0.5/nu - 0.25/gamma

        denom_a = alpha0**2 * (alpha0 + 1)
        denom_b = beta0**2 * (beta0 + 1)

        def compute_sigma(params, denom):
            params_outer = tf.reshape(params, (-1, 1)) * tf.reshape(params, (1, -1))
            params_sum = tf.reduce_sum(params)
            sigma = -params_outer / denom
            sigma = sigma + tf.linalg.diag(params * params_sum / denom)
            return sigma
        
        Sigma_d = compute_sigma(alpha, denom_a)
        Sigma_t = compute_sigma(beta, denom_b)

        c = tf.shape(alpha)[0]
        zeros_cc = tf.zeros((c, c), dtype=tf.float32)

        Sigma = tf.concat([
            tf.concat([Sigma_d, zeros_cc], axis=1),
            tf.concat([zeros_cc, Sigma_t], axis=1)
        ], axis=0)

        H = tf.concat([
            tf.concat([tf.linalg.diag(a_diag), tf.linalg.diag(b_diag)], axis=1),
            tf.concat([tf.linalg.diag(b_diag), tf.linalg.diag(c_diag)], axis=1)
        ], axis=0)
        SH = tf.matmul(Sigma, H)

        term1 = tf.reduce_sum(g * tf.squeeze(tf.matmul(Sigma, tf.reshape(g, (-1, 1)))))
        term2 = 0.5 * tf.linalg.trace(tf.matmul(SH, SH))
        return term1 + term2

    @staticmethod
    def random_acr(alpha, beta, min_beta=3.):
        mean = ApxC_LDL.jsd_dirichlet_expectation(alpha, beta)
        var = ApxC_LDL.jsd_dirichlet_variance(alpha, beta)

        mean = mean / np.log(2)
        var = var / (np.log(2) * np.log(2))

        var_max = mean * (1 - mean)**2 / (min_beta + 1 - mean)
        var = tf.minimum(var, var_max)

        nu = mean * (1 - mean) / var - 1.0
        alpha_ = mean * nu
        beta_ = (1 - mean) * nu

        def f(delta):
            return tf.math.betainc(alpha_, beta_, delta / np.log(2))

        init = Delta_LDL._simpson(f, 0, np.log(2))
        res = Delta_LDL._asr(f, 0, np.log(2), EPS, init, 5)
        res /= np.log(2)

        return res

    @staticmethod
    def acr(D, D_pred):
        js = ApxC_LDL._js_divergence(D, D_pred)
        def f(delta):
            return tf.reduce_mean(keras.activations.sigmoid(js - delta))
        init = Delta_LDL._simpson(f, 0, np.log(2))
        res = Delta_LDL._asr(f, 0, np.log(2), EPS, init, 5)
        res /= np.log(2)
        return res

    @staticmethod
    @tf.function
    def inv_digamma(y):
        return tf.where(y >= -2.22, tf.exp(y) + .5, -1. / (y + EULER_GAMMA))

    @staticmethod
    @tf.function
    def estimate_alpha(D, max_iter=100, tol=1e-7):
        log_p_bar = tf.reduce_mean(tf.math.log(D), axis=0)
        alpha = tf.ones(tf.shape(D)[1], dtype=tf.float32)
        i = tf.constant(0)

        def cond(i, _):
            return i < max_iter

        def body(i, alpha):
            alpha0 = tf.reduce_sum(alpha)
            y = tf.math.digamma(alpha0) + log_p_bar
            alpha_new = ApxC_LDL.inv_digamma(y)
            diff = tf.reduce_max(tf.abs(alpha_new - alpha))
            alpha = tf.clip_by_value(tf.where(diff < tol, alpha, alpha_new), EPS, 1e7)
            return i + 1, alpha

        _, alpha = tf.while_loop(cond, body, [i, alpha])

        return alpha

    def _loss(self, X, D, *_):
        D_pred = self._model(X)
        beta = ApxC_LDL.estimate_alpha(D_pred)

        self._l_up = ApxC_LDL.acr(D, D_pred)
        self._l_down = 1 - ApxC_LDL.random_acr(self._alpha_, beta)

        return self._l_up - self._lambda * self._l_down

    def train_step(self, *args, **kwargs):
        super().train_step(*args, **kwargs)
        self._lambda = self._l_up / (self._l_down + EPS)

    def _before_train(self):
        self._lambda = 1.
        self._n_dirichlet_samples = 100
        self._alpha_ = ApxC_LDL.estimate_alpha(self._D)
