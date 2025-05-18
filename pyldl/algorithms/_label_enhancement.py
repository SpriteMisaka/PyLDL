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

from pyldl.algorithms.base import BaseLE, BaseDeepLE, BaseGD, BaseAdam, BaseBFGS
from pyldl.algorithms.utils import pairwise_cosine


class FCM(BaseLE):
    """:class:`FCM <pyldl.algorithms.FCM>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, L, n_clusters=50, beta=2):
        super().fit(X, L)
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            self._X.T, n_clusters, beta,
            error=1e-7, maxiter=10000, init=None
        )
        A = np.matmul(L.T, u.T)
        D = fuzz.maxprod_composition(u.T, A.T)
        self._D = softmax(D, axis=1)
        return self


class KM(BaseLE):
    """:class:`KM <pyldl.algorithms.KM>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, L):
        super().fit(X, L)

        L = L > 0
        gamma = 1. / (2. * np.mean(pdist(self._X)) ** 2)
        s2 = np.zeros(self._L.shape)
        for j in range(self._n_outputs):
            c = self._X[L[:, j].reshape(-1)]
            temp1 = np.sum(rbf_kernel(c, gamma=gamma)) / (c.shape[0] ** 2)
            temp2 = -2 * np.sum(rbf_kernel(self._X, c, gamma=gamma), axis=1) / c.shape[0]
            s2[:, j] += temp1 + temp2 + 1

        r2 = np.max(s2, axis=0).reshape(1, -1)
        D = 1 - np.sqrt(s2 / (r2 + 1e-7))
        D *= self._L
        self._D = softmax(D, axis=1)
        return self


class LP(BaseLE):
    """:class:`LP <pyldl.algorithms.LP>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, L, epochs=500, alpha=.5):
        super().fit(X, L)

        dis = squareform(pdist(self._X, 'euclidean'))
        A = np.exp(- dis ** 2 / 2)
        temp = np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))
        P = np.matmul(np.matmul(temp, A), temp)

        D = self._L
        for _ in range(epochs):
            D = alpha * np.matmul(P, D) + (1 - alpha) * self._L
        self._D = softmax(D, axis=1)
        return self


class ML(BaseLE):
    """:class:`ML <pyldl.algorithms.ML>` is proposed in paper :cite:`2018:xu`.
    """

    def fit(self, X, L, beta=1):
        super().fit(X, L)
        L[L == 0] = -1
        knn = NearestNeighbors(n_neighbors=self.n_outputs+1)
        knn.fit(self._X)

        W = barycenter_kneighbors_graph(knn, self.n_outputs)
        W = W.toarray().astype(np.float32)
        W[np.isnan(W)] = 0
        W[np.isinf(W)] = 0

        M = np.matmul((np.eye(*W.shape) - W).T, (np.eye(*W.shape) - W))
        M += 1e-5 * np.eye(*M.shape)
        M = M.astype(np.float64)

        b = np.zeros((L.shape[0], 1), dtype=np.float64) - beta
        mu = np.zeros(L.shape)

        for k in range(self.n_outputs):
            A = -np.diag(L[:, k]).astype(np.float64)
            mu[:, k] = solve_qp(P=2*M,
                                q=np.zeros((L.shape[0],), dtype=np.float64),
                                G=A,
                                h=b,
                                solver='quadprog')

        self._D = softmax(mu, axis=1)
        return self


class GLLE(BaseBFGS, BaseDeepLE):
    """:class:`GLLE <pyldl.algorithms.GLLE>` is proposed in paper :cite:`2018:xu`.

    See also:

    .. bibliography:: bib/le/references.bib
        :filter: False
        :labelprefix: GLLE-
        :keyprefix: glle-

        2021:xu
    """

    def __init__(self, alpha=1e-2, beta=1e-4, sigma=1., **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma

    @tf.function
    def _E_loss(self, D):
        E_loss = 0.
        groups = tf.dynamic_partition(D, tf.constant(self._cluster_results), self._n_clusters)
        for i in range(self._n_clusters):
            E_loss += tf.linalg.trace(
                tf.transpose(groups[i]) @ self._E[i] @ tf.transpose(self._E[i]) @ groups[i]
            )
        return E_loss

    def _loss(self, params_1d):
        with tf.GradientTape() as tape:
            D = self._P @ self._params2model(params_1d)[0]
            E_loss = self._E_loss(D)
        E_gradients = tape.gradient(E_loss, self._E)
        self._E_optimizer.apply_gradients(zip(E_gradients, self._E))

        for i in range(self._n_clusters):
            Ei_norm = tf.linalg.norm(self._E[i], axis=1, keepdims=True)
            self._E[i].assign(self._E[i] / Ei_norm)

        D = self._P @ self._params2model(params_1d)[0]
        mse = tf.reduce_sum((self._L - D)**2)
        lap = tf.linalg.trace(tf.transpose(D) @ self._G @ D)
        return mse + self._alpha * lap + self._beta * self._E_loss(D)

    @staticmethod
    def _construct_P(X):
        gamma = 1. / (2. * np.mean(pdist(X)) ** 2)
        return tf.cast(rbf_kernel(X, gamma=gamma), dtype=tf.float32)

    def _before_train(self):
        self._P = self._construct_P(self._X)

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
        return self.get_2layer_model(self._P.shape[1], self._n_outputs, activation=None)

    def transform(self, X=None, L=None):
        P = self._P if X is None else self._construct_P(X)
        return keras.activations.softmax(self._call(P)).numpy()


class LEVI(BaseAdam, BaseDeepLE):
    """:class:`LEVI <pyldl.algorithms.LEVI>` is proposed in paper :cite:`2020:xu3`.

    See also:

    .. bibliography:: bib/le/references.bib
        :filter: False
        :labelprefix: LEVI-
        :keyprefix: levi-

        2023:xu4
    """

    def __init__(self, alpha=1., **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha

    def _call(self, X, L, transform=False):
        inputs = tf.concat((X, L), axis=1)

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
        L_hat = self._model["decoder_L"](samples)

        return d, std_d, samples, X_hat, L_hat

    def _loss(self, X, L, start, end):
        d, std_d, samples, X_hat, L_hat = self._call(X, L)
        kl = tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1)
        rec_X = keras.losses.mean_squared_error(X, X_hat)
        rec_L = keras.losses.binary_crossentropy(L, L_hat)

        return tf.reduce_sum((L - samples)**2) + self._alpha * tf.math.reduce_sum(kl + rec_X + rec_L)

    def _get_default_model(self):
        encoder = self.get_3layer_model(self._n_features + self._n_outputs, self._n_hidden, self._n_outputs*2,
                                        hidden_activation='softplus', output_activation=None)
        decoder_X = self.get_3layer_model(self._n_outputs, self._n_hidden, self._n_features,
                                          hidden_activation='softplus', output_activation=None)
        decoder_L = self.get_3layer_model(self._n_outputs, self._n_hidden, self._n_outputs,
                                          hidden_activation='softplus', output_activation=None)
        return {"encoder": encoder, "decoder_X": decoder_X, "decoder_L": decoder_L}

    def transform(self, X=None, L=None):
        if X is None and L is None:
            X, L = self._X, self._L
        return keras.activations.softmax(self._call(X, L, transform=True)).numpy()


class LIBLE(BaseAdam, BaseDeepLE):
    """:class:`LIBLE <pyldl.algorithms.LIBLE>` is proposed in paper :cite:`2023:zheng`.
    """

    def __init__(self, n_hidden=64, n_latent=64, alpha=1e-3, beta=1e-3, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        self._alpha = alpha
        self._beta = beta

    def _call(self, X, transform=False):
        latent = self._model["encoder"](X)
        mean = latent[:, :self._n_latent]
        if transform:
            return self._model["decoder_D"](mean)

        var = tf.math.softplus(latent[:, self._n_latent:])

        d = tfp.distributions.Normal(loc=mean, scale=var)
        std_d = tfp.distributions.Normal(loc=np.zeros(self._n_latent, dtype=np.float32),
                                         scale=np.ones(self._n_latent, dtype=np.float32))

        h = d.sample()
        L_hat = self._model["decoder_L"](h)
        D_hat = self._model["decoder_D"](h)
        g = self._model["decoder_g"](h)

        return d, std_d, L_hat, D_hat, g

    def _loss(self, X, L, start, end):
        d, std_d, L_hat, D_hat, g = self._call(X)
        kl = tf.reduce_sum(tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1))
        rec_L = tf.reduce_sum((L - L_hat)**2)
        rec_D = tf.reduce_sum(g**(-2) * (L - D_hat)**2 + tf.math.log(tf.abs(g**2)))

        return rec_L + self._alpha * kl + self._beta * rec_D

    def _get_default_model(self):
        encoder = self.get_3layer_model(self._n_features, self._n_hidden, self._n_latent*2,
                                        hidden_activation='tanh', output_activation=None)
        decoder_g = self.get_3layer_model(self._n_latent, self._n_hidden, 1,
                                          hidden_activation='tanh', output_activation='sigmoid')
        decoder_L = self.get_3layer_model(self._n_latent, self._n_hidden, self._n_outputs,
                                          hidden_activation='tanh', output_activation=None)
        decoder_D = self.get_3layer_model(self._n_latent, self._n_hidden, self._n_outputs,
                                            hidden_activation='tanh', output_activation=None)
        return {"encoder": encoder, "decoder_g": decoder_g, "decoder_L": decoder_L, "decoder_D": decoder_D}

    def transform(self, X=None, L=None):
        X = self._X if X is None else X
        return keras.activations.softmax(self._call(X, transform=True)).numpy()


class ConLE(BaseGD, BaseDeepLE):
    """:class:`ConLE <pyldl.algorithms.ConLE>` is proposed in paper :cite:`2023:wang4`.
    """

    def __init__(self, n_hidden=64, n_latent=64, alpha=1e-3, beta=1e-3,
                 tau=.5, threshold=.05, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        self._alpha = alpha
        self._beta = beta
        self._tau = tau
        self._threshold = threshold

    def _call(self, X, L, transform=False):
        Z = self._model["encoder_Z"](X)
        Q = self._model["encoder_Q"](L)
        H = tf.concat((Z, Q), axis=1)
        D = self._model["decoder"](H)
        return D if transform else (Z, Q, D)

    @staticmethod
    @tf.function
    def _con(X, Y, tau):
        C = tf.exp(pairwise_cosine(X, Y) / tau)
        CX = tf.exp((pairwise_cosine(X, X) - tf.eye(X.shape[0])) / tau)
        n = tf.shape(X)[0]
        con = 0.
        for i in range(n):
            numerator = C[i, i]
            denominator = tf.reduce_sum(CX[i]) + tf.reduce_sum(C[i]) - C[i, i]
            con += -tf.math.log(numerator / denominator)
        return con / tf.cast(n, tf.float32)

    @tf.function
    def _loss(self, X, L, start, end):
        Z, Q, D = self._call(X, L)
        con = self._con(Z, Q, self._tau) + self._con(Q, Z, self._tau)
        dis = tf.reduce_sum((L - D)**2)
        thr = tf.reduce_mean(tf.maximum(
            tf.reduce_max(D * (1 - L), axis=1) - tf.reduce_min(D * L + 1 - L, axis=1) + self._threshold, 0)
        )
        return con + self._alpha * dis + self._beta * thr

    def _get_default_model(self):
        encoder_Z = self.get_3layer_model(self._n_features, self._n_hidden, self._n_latent,
                                          hidden_activation=keras.layers.LeakyReLU(alpha=0.01), output_activation=None)
        encoder_Q = self.get_3layer_model(self._n_outputs, self._n_hidden, self._n_latent,
                                          hidden_activation=keras.layers.LeakyReLU(alpha=0.01), output_activation=None)
        decoder = self.get_3layer_model(self._n_latent*2, self._n_hidden, self._n_outputs,
                                        hidden_activation=keras.layers.LeakyReLU(alpha=0.01), output_activation='softmax')
        return {"encoder_Z": encoder_Z, "encoder_Q": encoder_Q, "decoder": decoder}

    def transform(self, X=None, L=None):
        if X is None and L is None:
            X, L = self._X, self._L
        return keras.activations.softmax(self._call(X, L, transform=True)).numpy()
