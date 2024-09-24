import numpy as np

import skfuzzy as fuzz
from qpsolvers import solve_qp

from scipy.special import softmax
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

import keras
import tensorflow as tf
import tensorflow_probability as tfp

from pyldl.algorithms.base import BaseLE, BaseDeepLE, BaseAdam, BaseBFGS


class FCM(BaseLE):
    """:class:`FCM <pyldl.algorithms.FCM>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, l, n_clusters=50, beta=2):
        super().fit(X, l)
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            self._X.T, n_clusters, beta,
            error=1e-7, maxiter=10000, init=None
        )
        A = np.matmul(l.T, u.T)
        y = fuzz.maxprod_composition(u.T, A.T)
        self._y = softmax(y, axis=1)
        return self


class KM(BaseLE):
    """:class:`KM <pyldl.algorithms.KM>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, l):
        super().fit(X, l)

        l = l > 0
        gamma = 1. / (2. * np.mean(pdist(self._X)) ** 2)
        s2 = np.zeros(self._l.shape)
        for j in range(self._n_outputs):
            c = self._X[l[:, j].reshape(-1)]
            temp1 = np.sum(rbf_kernel(c, gamma=gamma)) / (c.shape[0] ** 2)
            temp2 = -2 * np.sum(rbf_kernel(self._X, c, gamma=gamma), axis=1) / c.shape[0]
            s2[:, j] += temp1 + temp2 + 1

        r2 = np.max(s2, axis=0).reshape(1, -1)
        y = 1 - np.sqrt(s2 / (r2 + 1e-7))
        y *= self._l
        self._y = softmax(y, axis=1)
        return self


class LP(BaseLE):
    """:class:`LP <pyldl.algorithms.LP>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, l, epochs=500, alpha=.5):
        super().fit(X, l)

        dis = squareform(pdist(self._X, 'euclidean'))
        A = np.exp(- dis ** 2 / 2)
        temp = np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))
        P = np.matmul(np.matmul(temp, A), temp)

        y = self._l
        for _ in range(epochs):
            y = alpha * np.matmul(P, y) + (1 - alpha) * self._l
        self._y = softmax(y, axis=1)
        return self


class ML(BaseLE):
    """:class:`ML <pyldl.algorithms.ML>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, l, beta=1):
        super().fit(X, l)
        l[l == 0] = -1
        knn = NearestNeighbors(n_neighbors=self.n_outputs+1)
        knn.fit(self._X)

        W = barycenter_kneighbors_graph(knn, self.n_outputs)
        W = W.toarray().astype(np.float32)
        W[np.isnan(W)] = 0
        W[np.isinf(W)] = 0

        M = np.matmul((np.eye(*W.shape) - W).T, (np.eye(*W.shape) - W))
        M += 1e-5 * np.eye(*M.shape)
        M = M.astype(np.float64)

        b = np.zeros((l.shape[0], 1), dtype=np.float64) - beta
        mu = np.zeros(l.shape)

        for k in range(self.n_outputs):
            A = -np.diag(l[:, k]).astype(np.float64)
            mu[:, k] = solve_qp(P=2*M,
                                q=np.zeros((l.shape[0],), dtype=np.float64),
                                G=A,
                                h=b,
                                solver='quadprog')

        self._y = softmax(mu, axis=1)
        return self


class GLLE(BaseBFGS, BaseDeepLE):
    """:class:`GLLE <pyldl.algorithms.GLLE>` is proposed in paper :cite:`2018:xu`.

    See also:

    .. bibliography:: le_references.bib
        :filter: False
        :labelprefix: GLLE-
        :keyprefix: glle-

        2021:xu
    """

    @tf.function
    def _E_loss(self, y):
        E_loss = 0.
        groups = tf.dynamic_partition(y, tf.constant(self._cluster_results), self._n_clusters)
        for i in range(self._n_clusters):
            E_loss += tf.linalg.trace(
                tf.transpose(groups[i]) @ self._E[i] @ tf.transpose(self._E[i]) @ groups[i]
            )
        return E_loss

    def _loss(self, params_1d):
        with tf.GradientTape() as tape:
            y = self._P @ self._params2model(params_1d)[0]
            E_loss = self._E_loss(y)
        E_gradients = tape.gradient(E_loss, self._E)
        self._E_optimizer.apply_gradients(zip(E_gradients, self._E))

        for i in range(self._n_clusters):
            Ei_norm = tf.linalg.norm(self._E[i], axis=1, keepdims=True)
            self._E[i].assign(self._E[i] / Ei_norm)

        y = self._P @ self._params2model(params_1d)[0]
        mse = tf.reduce_sum((self._l - y)**2)
        lap = tf.linalg.trace(tf.transpose(y) @ self._G @ y)
        return mse + self._alpha * lap + self._beta * self._E_loss(y)

    def _before_train(self):
        gamma = 1. / (2. * np.mean(pdist(self._X)) ** 2)
        self._P = tf.cast(rbf_kernel(self._X, gamma=gamma), dtype=tf.float32)
        
        self._nn = NearestNeighbors(n_neighbors=self._n_outputs+1)
        self._nn.fit(self._X)
        graph = self._nn.kneighbors_graph()

        A = tf.exp(-(cdist(self._X, self._X) ** 2) / (2 * self._sigma ** 2))
        A = tf.cast(A * graph.toarray(), dtype=tf.float32)
        A_hat = tf.linalg.diag(tf.reduce_sum(A, axis=1))

        self._G = tf.cast(A_hat - A, dtype=tf.float32)

        k_means = KMeans()
        self._cluster_results = k_means.fit_predict(self._X)
        self._n_clusters = k_means.n_clusters
        cluster_counts = tf.math.bincount(self._cluster_results)

        self._E = [tf.Variable(tf.random.normal((cluster_counts[i], cluster_counts[i])),
                               trainable=True) for i in range(self._n_clusters)]
        self._E_optimizer = keras.optimizers.SGD()

    def _get_default_model(self):
        return self.get_2layer_model(self._P.shape[1], self._n_outputs, softmax=False)

    def fit(self, X, l, alpha=1e-2, beta=1e-4, sigma=1., max_iterations=50, **kwargs):
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        return super().fit(X, l, **kwargs)

    def transform(self):
        return keras.activations.softmax(self._call(self._P)).numpy()


class LEVI(BaseAdam, BaseDeepLE):
    """:class:`LEVI <pyldl.algorithms.LEVI>` is proposed in paper :cite:`2020:xu3`.

    See also:

    .. bibliography:: le_references.bib
        :filter: False
        :labelprefix: LEVI-
        :keyprefix: levi-

        2023:xu4
    """

    def _call(self, X, l, transform=False):
        inputs = tf.concat((X, l), axis=1)

        latent = self._model["encoder"](inputs)
        mean = latent[:, :self._n_outputs]
        if transform:
            return mean

        var = tf.math.softplus(latent[:, self._n_outputs:])

        d = tfp.distributions.Normal(loc=mean, scale=var)
        std_d = tfp.distributions.Normal(loc=np.zeros(self._n_outputs, dtype=np.float32),
                                         scale=np.ones(self._n_outputs, dtype=np.float32))

        samples = d.sample()
        X_hat = self._model["decoder_X"](samples)
        l_hat = self._model["decoder_l"](samples)

        return d, std_d, samples, X_hat, l_hat

    def _loss(self, X, l, start, end):
        d, std_d, samples, X_hat, l_hat = self._call(X, l)
        kl = tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1)
        rec_X = keras.losses.mean_squared_error(X, X_hat)
        rec_y = keras.losses.binary_crossentropy(l, l_hat)

        return tf.reduce_sum((l - samples)**2) + self._alpha * tf.math.reduce_sum(kl + rec_X + rec_y)

    def _get_default_model(self):

        input_shape = self._n_features + self._n_outputs

        encoder = keras.Sequential([keras.layers.InputLayer(input_shape=input_shape),
                                    keras.layers.Dense(self._n_hidden, activation='softplus'),
                                    keras.layers.Dense(self._n_outputs*2, activation=None)])

        decoder_X = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_outputs),
                                      keras.layers.Dense(self._n_hidden, activation='softplus'),
                                      keras.layers.Dense(self._n_features, activation=None)])

        decoder_l = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_outputs),
                                      keras.layers.Dense(self._n_hidden, activation='softplus'),
                                      keras.layers.Dense(self._n_outputs, activation=None)])

        return {"encoder": encoder, "decoder_X": decoder_X, "decoder_l": decoder_l}

    def fit(self, X, l, alpha=1., **kwargs):
        self._alpha = alpha
        return super().fit(X, l, **kwargs)

    def transform(self):
        return keras.activations.softmax(
            self._call(self._X, self._l, transform=True)
        ).numpy()


class LIBLE(BaseAdam, BaseDeepLE):
    """:class:`LIBLE <pyldl.algorithms.LIBLE>` is proposed in paper :cite:`2023:zheng`.
    """

    def __init__(self, n_hidden=64, n_latent=64, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _call(self, X, transform=False):
        latent = self._model["encoder"](X)
        mean = latent[:, :self._n_latent]
        if transform:
            return self._model["decoder_y"](mean)

        var = tf.math.softplus(latent[:, self._n_latent:])

        d = tfp.distributions.Normal(loc=mean, scale=var)
        std_d = tfp.distributions.Normal(loc=np.zeros(self._n_latent, dtype=np.float32),
                                         scale=np.ones(self._n_latent, dtype=np.float32))

        h = d.sample()
        l_hat = self._model["decoder_l"](h)
        y_hat = self._model["decoder_y"](h)
        g = self._model["decoder_g"](h)

        return d, std_d, l_hat, y_hat, g

    def _loss(self, X, l, start, end):
        d, std_d, l_hat, y_hat, g = self._call(X)
        kl = tf.reduce_sum(tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1))
        rec_l = tf.reduce_sum((l - l_hat)**2)
        rec_y = tf.reduce_sum(g**(-2) * (l - y_hat)**2 + tf.math.log(tf.abs(g**2)))

        return rec_l + self._alpha * kl + self._beta * rec_y

    def _get_default_model(self):

        encoder = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                    keras.layers.Dense(self._n_hidden, activation='tanh'),
                                    keras.layers.Dense(self._n_latent*2, activation=None)])

        decoder_g = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_latent),
                                      keras.layers.Dense(self._n_hidden, activation='tanh'),
                                      keras.layers.Dense(1, activation='sigmoid')])
        
        decoder_l = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_latent),
                                      keras.layers.Dense(self._n_hidden, activation='tanh'),
                                      keras.layers.Dense(self._n_outputs, activation=None)])
        
        decoder_y = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_latent),
                                      keras.layers.Dense(self._n_hidden, activation='tanh'),
                                      keras.layers.Dense(self._n_outputs, activation=None)])

        return {"encoder": encoder, "decoder_g": decoder_g, "decoder_l": decoder_l, "decoder_y": decoder_y}

    def fit(self, X, l, alpha=1e-3, beta=1e-3, **kwargs):
        self._alpha = alpha
        self._beta = beta
        return super().fit(X, l, **kwargs)

    def transform(self):
        return keras.activations.softmax(self._call(self._X, transform=True)).numpy()
