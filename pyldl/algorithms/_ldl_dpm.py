import numpy as np

import keras
import tensorflow as tf

from scipy.special import softplus

from pyldl.algorithms.base import BaseLDL

EPS = np.finfo(np.float64).eps


class LDL_DPM(BaseLDL):
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
            acc *= (1. - vk)
        self._w = np.array(w, dtype=np.float32)
        self._w_tf = tf.constant(self._w, dtype=tf.float32)

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

    def _extend_cluster(self):
        if self.k >= self.max_k:
            return
        self.k += 1
        self._set_v(np.append(self._v, np.array(
            np.random.beta(1., self.concentration), dtype=np.float32
        )))

        theta_new = tf.Variable(tf.random.normal(
            [1, self._n_features + 1, self._n_outputs], stddev=0.05
        ), dtype=tf.float32)
        gamma_new = tf.Variable(tf.zeros([self._n_features + 1, 1], dtype=tf.float32))

        self._Theta = tf.Variable(tf.concat([self._Theta, theta_new], axis=0))
        self._Gamma = tf.Variable(tf.concat([self._Gamma, gamma_new], axis=1))
        self._optimizer = keras.optimizers.Adam()

    def _Theta_and_Gamma_loss(self, X, D):
        X_aug = tf.concat([X, tf.ones([tf.shape(X)[0], 1])], axis=1)

        g = X_aug @ self._Gamma
        log_pi = tf.math.log(self._w_tf) + g
        pi = tf.nn.softmax(log_pi, axis=1)

        alpha_list = []
        for k in range(self.k):
            tmp = tf.math.softplus(X_aug @ self._Theta[k])
            tmp = tmp / tf.reduce_sum(tmp, axis=1, keepdims=True) * self.strength
            alpha_list.append(tmp)
        alpha = tf.stack(alpha_list, axis=1)

        safe_d = tf.expand_dims(D, axis=1)
        log_pdf = (tf.math.lgamma(tf.reduce_sum(alpha, axis=-1))
                   - tf.reduce_sum(tf.math.lgamma(alpha), axis=-1)
                   + tf.reduce_sum((alpha - 1.) * tf.math.log(safe_d + EPS), axis=-1))

        log_mix = tf.reduce_logsumexp(tf.math.log(pi) + log_pdf, axis=1)
        nll = -tf.reduce_mean(log_mix)

        reg = (self.alpha * tf.reduce_sum(tf.square(self._Theta))
               + self.beta * tf.reduce_sum(tf.square(self._Gamma)))

        return nll + reg

    @tf.function
    def _update_Theta_and_Gamma(self, X, D):
        with tf.GradientTape() as tape:
            loss = self._Theta_and_Gamma_loss(X, D)
        grads = tape.gradient(loss, [self._Theta, self._Gamma])
        self._optimizer.apply_gradients(zip(grads, [self._Theta, self._Gamma]))

    def _slice_gibbs(self, X, D, z, u):
        X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        g_np = X_aug @ self._Gamma.numpy()
        Theta_np = self._Theta.numpy()

        for i in range(X.shape[0]):
            u[i] = np.random.rand() * self._w[z[i]]
            feas = [k for k in range(self.k) if self._w[k] > u[i]]

            logw = []
            for k in feas:
                alpha = softplus(X_aug[i] @ Theta_np[k])
                alpha_sum = alpha.sum()
                if alpha_sum == 0.:
                    alpha = np.ones_like(alpha) * 1e-6
                    alpha_sum = alpha.sum()
                alpha = alpha / alpha_sum * self.strength
                alpha = np.clip(alpha, 1e-12, None)

                log_pdf = (np.log(alpha.sum())
                           - np.log(alpha).sum()
                           + ((alpha - 1.) * np.log(D[i] + 1e-12)).sum())
                logw.append(g_np[i, k] + log_pdf)

            logw = np.array(logw)
            p = np.exp(logw - logw.max())
            if p.sum() == 0.:
                p = np.ones_like(p) / len(p)
            else:
                p /= p.sum()

            z[i] = np.random.choice(feas, p=p)

        return z, u

    def predict(self, X):
        X_aug = tf.concat([X, tf.ones([tf.shape(X)[0], 1])], axis=1)

        log_pi = tf.math.log(self._w_tf) + X_aug @ self._Gamma
        pi = tf.nn.softmax(log_pi, axis=1)

        mu_list = []
        for k in range(self.k):
            tmp = tf.math.softplus(X_aug @ self._Theta[k])
            tmp = tmp / tf.reduce_sum(tmp, axis=1, keepdims=True)
            mu_list.append(tmp)
        mu = tf.stack(mu_list, axis=1)
        return tf.reduce_sum(tf.expand_dims(pi, -1) * mu, axis=1).numpy()

    def fit(self, X, D, *, outer_iterations=100, inner_iterations=100, batch_size=None):
        super().fit(X, D)
        if batch_size is None:
            batch_size = self._n_samples

        self.k = 1
        self._set_v(np.random.beta(1., self.concentration, size=1).astype(np.float32))

        self._Theta = tf.Variable(tf.random.normal(
            [1, self._n_features + 1, self._n_outputs], stddev=0.05
        ), dtype=tf.float32)
        self._Gamma = tf.Variable(tf.zeros(
            [self._n_features + 1, 1], dtype=tf.float32
        ), dtype=tf.float32)

        z = np.zeros(self._n_samples, dtype=np.int32)
        u = np.random.rand(self._n_samples) * self._w[0]

        X_tf = tf.constant(X, dtype=tf.float32)
        D_tf = tf.constant(D, dtype=tf.float32)
        self._optimizer = keras.optimizers.Adam()
        for _ in range(outer_iterations):
            self._update_v(z, u)
            z, u = self._slice_gibbs(self._X, self._D, z, u)
            for _ in range(inner_iterations):
                for start in range(0, self._n_samples, batch_size):
                    end = min(start + batch_size, self._n_samples)
                    self._update_Theta_and_Gamma(X_tf[start:end], D_tf[start:end])
