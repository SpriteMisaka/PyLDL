import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseIter, BaseLDL
from pyldl.algorithms.utils import gammaln


EPS = np.finfo(np.float64).eps


class _DPMParameters(keras.Model):

    def __init__(self, theta, gamma, strength, alpha, beta):
        super().__init__(name='dpm_parameters')
        self.strength = strength
        self.alpha = alpha
        self.beta = beta
        self.Theta = self.add_weight(
            name='Theta', shape=theta.shape,
            initializer=keras.initializers.Constant(theta), trainable=True
        )
        self.Gamma = self.add_weight(
            name='Gamma', shape=gamma.shape,
            initializer=keras.initializers.Constant(gamma), trainable=True
        )

    def call(self, inputs):
        X, D, w = inputs
        X_aug = ops.concatenate([X, ops.ones((ops.shape(X)[0], 1))], axis=1)
        g = X_aug @ self.Gamma
        pi = keras.activations.softmax(ops.log(w) + g, axis=1)
        alpha_values = []
        for k in range(self.Theta.shape[0]):
            value = ops.softplus(X_aug @ self.Theta[k])
            value = value / ops.sum(value, axis=1, keepdims=True) * self.strength
            alpha_values.append(value)
        dirichlet_alpha = ops.stack(alpha_values, axis=1)
        safe_d = ops.expand_dims(D, axis=1)
        log_pdf = (
            gammaln(ops.sum(dirichlet_alpha, axis=-1))
            - ops.sum(gammaln(dirichlet_alpha), axis=-1)
            + ops.sum((dirichlet_alpha - 1.) * ops.log(safe_d + EPS), axis=-1)
        )
        log_mix = ops.logsumexp(ops.log(pi) + log_pdf, axis=1)
        nll = -ops.mean(log_mix)
        reg = self.alpha * ops.sum(ops.square(self.Theta)) + self.beta * ops.sum(ops.square(self.Gamma))
        self.add_loss(nll + reg)
        return nll + reg


class LDL_DPM(BaseIter, BaseLDL):
    """:class:`LDL-DPM <pyldl.algorithms.LDL_DPM>` is proposed in paper :cite:`2026:wang`.
    DPM refers to *Dirichlet process mixture* (model).
    """

    def __init__(self, concentration=1., strength=3., max_k=15,
                 alpha=1e-6, beta=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.concentration = concentration
        self.strength = strength
        self.max_k = max_k
        self.alpha = alpha
        self.beta = beta

    def _update_w(self):
        w = []
        acc = 1.
        for vk in self._v:
            w.append(vk * acc)
            acc *= 1. - vk
        self._w = np.array(w, dtype=np.float32)
        self._w_tensor = ops.convert_to_tensor(self._w, dtype="float32")

    def _set_v(self, v):
        self._v = v
        self._update_w()

    def _update_v(self, z, u):
        counts = np.bincount(z, minlength=self.k)
        if self.k > 1:
            new_v = []
            for k in range(self.k):
                a = 1. + counts[k]
                b = self.concentration + counts[k + 1:].sum()
                new_v.append(np.random.beta(a, b))
            self._set_v(np.array(new_v, dtype=np.float32))
        if self._w.sum() < 1. - u.min():
            self._extend_cluster()

    def _build_parameters(self, theta, gamma):
        self._parameters = _DPMParameters(
            theta, gamma, self.strength, self.alpha, self.beta
        )
        self._parameters.compile(optimizer=keras.optimizers.Adam())
        self._Theta = self._parameters.Theta
        self._Gamma = self._parameters.Gamma

    def _extend_cluster(self):
        if self.k >= self.max_k:
            return
        self.k += 1
        self._set_v(np.append(
            self._v, np.array(np.random.beta(1., self.concentration), dtype=np.float32)
        ))
        theta = np.concatenate([
            ops.convert_to_numpy(self._Theta),
            np.random.normal(0., .05, (1, self._n_features + 1, self._n_outputs)).astype(np.float32)
        ], axis=0)
        gamma = np.concatenate([
            ops.convert_to_numpy(self._Gamma),
            np.zeros((self._n_features + 1, 1), dtype=np.float32)
        ], axis=1)
        self._build_parameters(theta, gamma)

    def _update_Theta_and_Gamma(self, X, D):
        return self._parameters.train_on_batch((X, D, self._w_tensor))

    def _slice_gibbs(self, X, D, z, u):
        from scipy.special import gammaln
        X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        g_np = X_aug @ ops.convert_to_numpy(self._Gamma)
        Theta_np = ops.convert_to_numpy(self._Theta)
        for i in range(X.shape[0]):
            u[i] = np.random.rand() * self._w[z[i]]
            feas = [k for k in range(self.k) if self._w[k] > u[i]]
            logw = []
            for k in feas:
                alpha = np.logaddexp(0., X_aug[i] @ Theta_np[k])
                alpha_sum = alpha.sum()
                if alpha_sum == 0.:
                    alpha = np.ones_like(alpha) * 1e-6
                    alpha_sum = alpha.sum()
                alpha = np.clip(alpha / alpha_sum * self.strength, 1e-12, None)
                log_pdf = (
                    gammaln(alpha.sum()) - gammaln(alpha).sum()
                    + ((alpha - 1.) * np.log(D[i] + 1e-12)).sum()
                )
                logw.append(g_np[i, k] + log_pdf)
            logw = np.array(logw)
            p = np.exp(logw - logw.max())
            p = np.ones_like(p) / len(p) if p.sum() == 0. else p / p.sum()
            z[i] = np.random.choice(feas, p=p)
        return z, u

    def predict(self, X):
        X = ops.convert_to_tensor(X, dtype="float32")
        X_aug = ops.concatenate([X, ops.ones((ops.shape(X)[0], 1))], axis=1)
        pi = keras.activations.softmax(ops.log(self._w_tensor) + X_aug @ self._Gamma, axis=1)
        mu = []
        for k in range(self.k):
            value = ops.softplus(X_aug @ self._Theta[k])
            mu.append(value / ops.sum(value, axis=1, keepdims=True))
        mu = ops.stack(mu, axis=1)
        return ops.convert_to_numpy(ops.sum(ops.expand_dims(pi, -1) * mu, axis=1))

    def _before_train(self, inner_iterations=100, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self._n_samples
        self._inner_iterations = inner_iterations
        self._batch_size = batch_size
        self.k = 1
        self._set_v(np.random.beta(1., self.concentration, size=1).astype(np.float32))
        theta = np.random.normal(
            0., .05, (1, self._n_features + 1, self._n_outputs)
        ).astype(np.float32)
        gamma = np.zeros((self._n_features + 1, 1), dtype=np.float32)
        self._build_parameters(theta, gamma)
        self._z = np.zeros(self._n_samples, dtype=np.int32)
        self._u = np.random.rand(self._n_samples) * self._w[0]
        self._X_tensor = ops.convert_to_tensor(self._X, dtype="float32")
        self._D_tensor = ops.convert_to_tensor(self._D, dtype="float32")

    def _run_iteration(self):
        self._update_v(self._z, self._u)
        self._z, self._u = self._slice_gibbs(
            self._X, self._D, self._z, self._u
        )
        losses = []
        for _ in range(self._inner_iterations):
            for start in range(0, self._n_samples, self._batch_size):
                end = min(start + self._batch_size, self._n_samples)
                loss = self._update_Theta_and_Gamma(
                    self._X_tensor[start:end], self._D_tensor[start:end]
                )
                losses.append(float(np.asarray(loss)))
        return float(np.mean(losses)), False

    def get_weights(self):
        return {
            'k': self.k,
            'v': self._v.copy(),
            'Theta': ops.convert_to_numpy(self._Theta).copy(),
            'Gamma': ops.convert_to_numpy(self._Gamma).copy(),
        }

    def set_weights(self, state):
        self.k = state['k']
        self._set_v(state['v'].copy())
        self._build_parameters(state['Theta'], state['Gamma'])
