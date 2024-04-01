import numpy as np

import skfuzzy as fuzz
from qpsolvers import solve_qp

from scipy.special import softmax
from scipy.spatial.distance import pdist, cdist

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

import keras
import tensorflow as tf
import tensorflow_probability as tfp

from pyldl.algorithms.base import BaseLE, BaseDeepLE, DeepBFGS


class FCM(BaseLE):

    def __init__(self, random_state=None):
        super().__init__(random_state)

    def fit_transform(self, X, l, n_clusters=50, beta=2):
        super().fit_transform(X, l)

        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            self._X.T, n_clusters, beta,
            error=1e-7, maxiter=10000, init=None
        )
        A = np.matmul(l.T, u.T)
        y = fuzz.maxprod_composition(u.T, A.T)
        return softmax(y, axis=1)


class KM(BaseLE):

    def __init__(self, random_state=None):
        super().__init__(random_state)

    def fit_transform(self, X, l):
        super().fit_transform(X, l)

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
        return softmax(y, axis=1)


class LP(BaseLE):

    def __init__(self, random_state=None):
        super().__init__(random_state)

    def fit_transform(self, X, l, epochs=3000, alpha=.5):
        super().fit_transform(X, l)

        dis = np.linalg.norm(self._X[:, None] - self._X, axis=-1)
        A = np.exp(- dis ** 2 / 2)
        temp = np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))
        P = np.matmul(np.matmul(temp, A), temp)

        y = self._l
        for _ in range(epochs):
            y = alpha * np.matmul(P, y) + (1 - alpha) * self._l
        y = softmax(y, axis=1)

        return y


class ML(BaseLE):

    def __init__(self, random_state=None):
        super().__init__(random_state)

    def fit_transform(self, X, l, beta=1):
        super().fit_transform(X, l)
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

        return softmax(mu, axis=1)


class GLLE(BaseDeepLE, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _get_obj_func(self, model, loss_function, P, l):

        def _f(params_1d):

            with tf.GradientTape() as tape:
                y = model(P)
                E_loss = self._E_loss(y)

            E_gradients = tape.gradient(E_loss, self._E)
            self._E_optimizer.apply_gradients(zip(E_gradients, self._E))

            for i in range(self._n_clusters):
                Ei_norm = tf.linalg.norm(self._E[i], axis=1, keepdims=True)
                self._E[i].assign(self._E[i] / Ei_norm)

            with tf.GradientTape() as tape:
                self._assign_new_model_parameters(params_1d, model)
                y = model(P)
                loss = loss_function(l, y)

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients = tf.dynamic_stitch(self._idx, gradients)

            return loss, gradients

        return _f
    
    @tf.function
    def _E_loss(self, y):
        E_loss = 0.
        groups = tf.dynamic_partition(y, tf.constant(self._cluster_results), self._n_clusters)
        for i in range(self._n_clusters):
            E_loss += tf.linalg.trace(
                tf.transpose(groups[i]) @ self._E[i] @ tf.transpose(self._E[i]) @ groups[i]
            )
        return E_loss

    @tf.function
    def _loss(self, l, y):
        mse = tf.reduce_sum((l - y)**2)
        lap = tf.linalg.trace(tf.transpose(y) @ self._G @ y)
        return mse + self._alpha * lap + self._beta * self._E_loss(y)

    def fit_transform(self, X, l, max_iterations=500, corr_learning_rate=1e-2,
                      alpha=1e-2, beta=1e-4, sigma=10.):
        super().fit_transform(X, l)

        self._alpha = alpha
        self._beta = beta

        self._sigma = sigma

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
        self._E_optimizer = keras.optimizers.SGD(learning_rate=corr_learning_rate)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._P.shape[1]),
             keras.layers.Dense(self._n_outputs, activation=None)])
        
        self._optimize_bfgs(self._model, self._loss, self._P, self._l, max_iterations)

        return keras.activations.softmax(self._model(self._P))


class LEVI(BaseDeepLE):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _loss(self, X, l):
        inputs = tf.concat((X, l), axis=1)

        latent = self._encoder(inputs)
        mean = latent[:, :self._n_outputs]
        var = tf.math.softplus(latent[:, self._n_outputs:])

        d = tfp.distributions.Normal(loc=mean, scale=var)
        std_d = tfp.distributions.Normal(loc=np.zeros(self._n_outputs, dtype=np.float32),
                                         scale=np.ones(self._n_outputs, dtype=np.float32))

        samples = d.sample()
        X_hat = self._decoder_X(samples)
        l_hat = self._decoder_l(samples)

        kl = tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1)
        rec_X = keras.losses.mean_squared_error(X, X_hat)
        rec_y = keras.losses.binary_crossentropy(l, l_hat)

        return tf.reduce_sum((l - samples)**2) + self._alpha * tf.math.reduce_sum(kl + rec_X + rec_y)

    def fit_transform(self, X, l, learning_rate=1e-4, epochs=1000, alpha=1.):
        super().fit_transform(X, l)

        self._alpha = alpha

        input_shape = self._n_features + self._n_outputs

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._encoder = keras.Sequential([keras.layers.InputLayer(input_shape=input_shape),
                                          keras.layers.Dense(self._n_hidden, activation='softplus'),
                                          keras.layers.Dense(self._n_outputs*2, activation=None)])

        self._decoder_X = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_outputs),
                                            keras.layers.Dense(self._n_hidden, activation='softplus'),
                                            keras.layers.Dense(self._n_features, activation=None)])
        
        self._decoder_l = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_outputs),
                                            keras.layers.Dense(self._n_hidden, activation='softplus'),
                                            keras.layers.Dense(self._n_outputs, activation=None)])

        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._l)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        inputs = tf.concat((self._X, self._l), axis=1)
        latent = self._encoder(inputs)
        mean = latent[:, :self._n_outputs].numpy()
        return softmax(mean, axis=1)


class LIBLE(BaseDeepLE):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _loss(self, X, l):
        latent = self._encoder(X)
        mean = latent[:, :self._n_latent]
        var = tf.math.softplus(latent[:, self._n_latent:])

        d = tfp.distributions.Normal(loc=mean, scale=var)
        std_d = tfp.distributions.Normal(loc=np.zeros(self._n_latent, dtype=np.float32),
                                         scale=np.ones(self._n_latent, dtype=np.float32))

        h = d.sample()
        l_hat = self._decoder_l(h)
        y_hat = self._decoder_y(h)
        g = self._decoder_g(h)

        kl = tf.reduce_sum(tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1))
        rec_l = tf.reduce_sum((l - l_hat)**2)
        rec_y = tf.reduce_sum(g**(-2) * (l - y_hat)**2 + tf.math.log(tf.abs(g**2)))

        return rec_l + self._alpha * kl + self._beta * rec_y

    def fit_transform(self, X, l, learning_rate=1e-3, epochs=1500, alpha=1e-3, beta=1.):
        super().fit_transform(X, l)

        self._alpha = alpha
        self._beta = beta

        if self._n_latent is None:
            self._n_latent = self._n_features
        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._encoder = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                          keras.layers.Dense(self._n_hidden, activation='tanh'),
                                          keras.layers.Dense(self._n_latent*2, activation=None)])

        self._decoder_g = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_latent),
                                            keras.layers.Dense(self._n_hidden, activation='tanh'),
                                            keras.layers.Dense(1, activation='sigmoid')])
        
        self._decoder_l = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_latent),
                                            keras.layers.Dense(self._n_hidden, activation='tanh'),
                                            keras.layers.Dense(self._n_outputs, activation=None)])
        
        self._decoder_y = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_latent),
                                            keras.layers.Dense(self._n_hidden, activation='tanh'),
                                            keras.layers.Dense(self._n_outputs, activation=None)])

        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._l)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        latent = self._encoder(X)
        mean = latent[:, :self._n_latent].numpy()
        return softmax(self._decoder_y(mean), axis=1)
