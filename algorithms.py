import os
import copy
import warnings

import numpy as np

from qpsolvers import solve_qp

from scipy.optimize import minimize, fsolve
from scipy.special import softmax
from scipy.spatial.distance import pdist

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVR
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

import skfuzzy as fuzz

import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from metrics import score, sort_loss
from rprop import RProp


class _Base():

    def __init__(self, random_state=None):
        if not random_state is None:
            np.random.seed(random_state)

        self._n_features = None
        self._n_outputs = None

    def fit(self, X, y=None):
        self._X = X
        self._y = y
        self._n_features = self._X.shape[1]
        if not self._y is None:
            self._n_outputs = self._y.shape[1]

    def _not_been_fit(self):
        raise ValueError("The model has not yet been fit. "
                         "Try to call 'fit()' first with some training data.")

    @property
    def n_features(self):
        if self._n_features is None:
            self._not_been_fit()
        return self._n_features

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            self._not_been_fit()
        return self._n_outputs


class _BaseLDL(_Base):

    def predict(self, _):
        pass

    def score(self, X, y, metrics=None):
        if metrics is None:
            metrics = ["chebyshev", "clark", "canberra", "kl_divergence",
                       "cosine", "intersection"]
        return score(y, self.predict(X), metrics=metrics)


class _BaseLE(_Base):

    def fit_transform(self, X, l):
        super().fit(X, None)
        self._l = l
        self._n_outputs = self._l.shape[1]

    def score(self, X, l, y, metrics=None):
        if metrics is None:
            metrics = ["chebyshev", "clark", "canberra", "kl_divergence",
                       "cosine", "intersection"]
        return score(y, self.fit_transform(X, l), metrics=metrics)


class BaseLDL(_BaseLDL, BaseEstimator):

    def __init__(self, random_state=None):
        super().__init__(random_state)


class BaseLE(_BaseLE, TransformerMixin, BaseEstimator):

    def __init__(self, random_state=None):
        super().__init__(random_state)


class _BaseDeep(keras.Model):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        keras.Model.__init__(self)
        if not random_state is None:
            tf.random.set_seed(random_state)
        self._n_latent = n_latent
        self._n_hidden = n_hidden

    def _l2_reg(self, model):
        reg = 0.
        for i in model.trainable_variables:
            reg += tf.reduce_sum(tf.square(i))
        return reg / 2.


class BaseDeepLDL(_BaseLDL, _BaseDeep):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        _BaseLDL.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit(self, X, y):
        _BaseLDL.fit(self, X, y)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._y = tf.cast(self._y, dtype=tf.float32)


class BaseDeepLE(_BaseLE, _BaseDeep):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        _BaseLE.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit_transform(self, X, l):
        _BaseLE.fit_transform(self, X, l)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._l = tf.cast(self._l, dtype=tf.float32)


class _SA(BaseLDL):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._W = None

    def _loss(self, y, y_pred):
        y_true = np.clip(y, 1e-7, 1)
        y_pred = np.clip(y_pred, 1e-7, 1)
        return -1 * np.sum(y_true * np.log(y_pred))

    def predict(self, X):
        return softmax(np.dot(X, self._W), axis=1)

    @property
    def W(self):
        if self._W is None:
            self._not_been_fit()
        return self._W


class SA_BFGS(_SA):

    def _obj_func(self, weights):
        W = weights.reshape(self._n_outputs, self._n_features).T
        y_pred = softmax(np.dot(self._X, W), axis=1)

        func_loss = self._loss(self._y, y_pred)
        func_grad = np.dot(self._X.T, y_pred - self._y).T.reshape(-1, )

        return func_loss, func_grad

    def fit(self, X, y, max_iterations=600, convergence_criterion=1e-6):
        super().fit(X, y)

        weights = np.random.uniform(-0.1, 0.1, self._n_features * self._n_outputs)

        optimize_result = minimize(self._obj_func, weights, method='L-BFGS-B', jac=True,
                                   options={'gtol': convergence_criterion,
                                            'disp': False, 'maxiter': max_iterations})
        
        self._W = optimize_result.x.reshape(self._n_outputs, self._n_features).T


class SA_IIS(_SA):

    def fit(self, X, y, max_iterations=600, convergence_criterion=1e-6):
        super().fit(X, y)
        warnings.filterwarnings('ignore', "The iteration is not making good progress,")

        weights = np.random.uniform(-0.1, 0.1, self._n_features * self._n_outputs)

        flag = True
        counter = 1
        W = weights.reshape(self._n_outputs, self._n_features).transpose()
        y_pred = softmax(np.dot(self._X, W), axis=1)

        while flag:
            delta = np.empty(shape=(self._X.shape[1], self._y.shape[1]), dtype=np.float32)
            for k in range(self._X.shape[1]):
                for j in range(self._y.shape[1]):
                    def func(x):
                        temp1 = np.sum(self._y[:, j] * self._X[:, k])
                        temp2 = np.sum(y_pred[:, j] * self._X[:, k] * \
                                       np.exp(x * np.sign(self._X[:, k]) * \
                                              np.sum(np.abs(self._X), axis=1)))
                        return temp1 - temp2
                    delta[k][j] = fsolve(func, .0, xtol=1e-10)

            l2 = self._loss(self._y, y_pred)
            weights += delta.transpose().ravel()
            y_pred = softmax(np.dot(self._X, W), axis=1)
            l1 = self._loss(self._y, y_pred)

            if l2 - l1 < convergence_criterion or counter >= max_iterations:
                flag = False

            W = weights.reshape(self._n_outputs, self._n_features).transpose()
            counter += 1

        self._W = W


class AA_KNN(BaseLDL):

    def __init__(self,
                 k=5,
                 random_state=None):

        super().__init__(random_state)

        self.k = k
        self._model = NearestNeighbors(n_neighbors=self.k)

    def fit(self, X, y):
        super().fit(X, y)
        self._model.fit(self._X)

    def predict(self, X):
        _, inds = self._model.kneighbors(X)
        return np.average(self._y[inds], axis=1)


class AA_BP(BaseDeepLDL):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss_function(self, y, y_pred):
        return tf.math.reduce_mean(keras.losses.mean_squared_error(y, y_pred))

    def fit(self, X, y, learning_rate=5e-3, epochs=3000,
            activation='sigmoid', optimizer='SGD'):
        super().fit(X, y)

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=(self._n_features,)),
                                        keras.layers.Dense(self._n_hidden, activation=activation),
                                        keras.layers.Dense(self._n_outputs, activation='softmax')])
        self._optimizer = eval(f'keras.optimizers.{optimizer}({learning_rate})')

        for _ in range(epochs):

            with tf.GradientTape() as tape:
                y_pred = self._model(self._X)
                loss = self._loss_function(self._y, y_pred)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict(self, X):
        return self._model(X)


class CAD(AA_BP):

    @tf.function
    def _loss_function(self, y, y_pred):
        def _CAD(y, y_pred):
            return tf.reduce_mean(tf.abs(
                tf.cumsum(y, axis=1) - tf.cumsum(y_pred, axis=1)
            ), axis=1)
        return tf.math.reduce_sum(
            tf.map_fn(lambda i: _CAD(y[:, :i], y_pred[:, :i]),
                      tf.range(1, self._n_outputs + 1),
                      fn_output_signature=tf.float32)
        )

    def fit(self, X, y, learning_rate=1e-4, epochs=500,
            activation='relu', optimizer='Adam'):
        return super().fit(X, y, learning_rate, epochs, activation, optimizer)


class QFD2(AA_BP):

    @tf.function
    def _loss_function(self, y, y_pred):
        Q = y - y_pred
        j = tf.reshape(tf.range(self._n_outputs), [self._n_outputs, 1])
        k = tf.reshape(tf.range(self._n_outputs), [1, self._n_outputs])
        A = tf.cast(1 - tf.abs(j - k) / (self._n_outputs - 1), dtype=tf.float32)
        return tf.math.reduce_mean(
            tf.linalg.diag_part(tf.matmul(tf.matmul(Q, A), tf.transpose(Q)))
        )

    def fit(self, X, y, learning_rate=1e-4, epochs=500,
            activation='relu', optimizer='Adam'):
        return super().fit(X, y, learning_rate, epochs, activation, optimizer)


class CJS(AA_BP):

    @tf.function
    def _loss_function(self, y, y_pred):
        def _CJS(y, y_pred):
            m = 0.5 * (y + y_pred)
            js = 0.5 * (keras.losses.kl_divergence(y, m) + keras.losses.kl_divergence(y_pred, m))
            return tf.reduce_mean(js)
        return tf.math.reduce_sum(
            tf.map_fn(lambda i: _CJS(y[:, :i], y_pred[:, :i]),
                      tf.range(1, self._n_outputs + 1),
                      fn_output_signature=tf.float32)
        )

    def fit(self, X, y, learning_rate=1e-4, epochs=500,
            activation='relu', optimizer='Adam'):
        return super().fit(X, y, learning_rate, epochs, activation, optimizer)


class CPNN(BaseDeepLDL):

    def _not_proper_mode(self):
        raise ValueError("The argument 'mode' can only be 'none', 'binary' or 'augment'.")

    def __init__(self, mode='none', v=5, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        if mode == 'none' or mode == 'binary' or mode == 'augment':
            self._mode = mode
        else:
            self._not_proper_mode()
        self._v = v

    def fit(self, X, y, learning_rate=5e-3, epochs=3000):
        super().fit(X, y)

        self._optimizer = RProp(init_alpha=learning_rate)

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        if self._mode == 'augment':
            one_hot = tf.one_hot(tf.math.argmax(self._y, axis=1), self.n_outputs)
            self._X = tf.repeat(self._X, self._v, axis=0)
            self._y = tf.repeat(self._y, self._v, axis=0)
            one_hot = tf.repeat(one_hot, self._v, axis=0)
            v = tf.reshape(tf.tile([1 / (i + 1) for i in range(self._v)], [X.shape[0]]), (-1, 1))
            self._y += self._y * one_hot * v

        input_shape = (self._n_features + (1 if self._mode == 'none' else self._n_outputs),)
        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=input_shape),
                                        keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                        keras.layers.Dense(1, activation=None)])

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.get_updates(self.trainable_variables, gradients)

    def _make_inputs(self, X):
        temp = tf.reshape(tf.tile([i + 1 for i in range(self._n_outputs)], [X.shape[0]]), (-1, 1))
        if self._mode != 'none':
            temp = OneHotEncoder(sparse=False).fit_transform(temp)
        return tf.concat([tf.cast(tf.repeat(X, self._n_outputs, axis=0), dtype=tf.float32),
                          tf.cast(temp, dtype=tf.float32)],
                          axis=1)

    def _call(self, X):
        inputs = self._make_inputs(X)
        outputs = self._model(inputs)
        results = tf.reshape(outputs, (X.shape[0], self._n_outputs))
        b = tf.reshape(-tf.math.log(tf.math.reduce_sum(tf.math.exp(results), axis=1)), (-1, 1))
        return tf.math.exp(b + results)

    def _loss(self, X, y):
        return tf.math.reduce_mean(keras.losses.kl_divergence(y, self._call(X)))

    def predict(self, X):
        return self._call(X)


class BCPNN(CPNN):

    def __init__(self, **params):
        super().__init__(mode='binary', **params)


class ACPNN(CPNN):

    def __init__(self, **params):
        super().__init__(mode='augment', **params)


class LDLF(BaseDeepLDL):

    def __init__(self, n_estimators=5, n_depth=6, n_hidden=None, n_latent=64, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)
        self._n_estimators = n_estimators
        self._n_depth = n_depth
        self._n_leaves = 2 ** n_depth

    def _call(self, X, i):
        decisions = tf.gather(self._model(X), self._phi[i], axis=1)
        decisions = tf.expand_dims(decisions, axis=2)
        decisions = tf.concat([decisions, 1 - decisions], axis=2)
        mu = tf.ones([X.shape[0], 1, 1])

        begin_idx = 1
        end_idx = 2

        for level in range(self._n_depth):
            mu = tf.reshape(mu, [X.shape[0], -1, 1])
            mu = tf.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [X.shape[0], self._n_leaves])

        return mu

    def fit(self, X, y, learning_rate=5e-2, epochs=3000):
        super().fit(X, y)

        self._phi = [np.random.choice(
            np.arange(self._n_latent), size=self._n_leaves, replace=False
        ) for _ in range(self._n_estimators)]

        self._pi = [tf.Variable(
            initial_value = tf.constant_initializer(1 / self.n_outputs)(
                shape=[self._n_leaves, self._n_outputs]
            ),
            dtype="float32", trainable=True,
        ) for _ in range(self._n_estimators)]

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                        keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                        keras.layers.Dense(self._n_latent, activation="sigmoid")])
        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as model_tape:
                loss = 0.
                for i in range(self._n_estimators):
                    _mu = self._call(X, i)
                    _prob = tf.matmul(_mu, self._pi[i])

                    loss += tf.math.reduce_mean(keras.losses.kl_divergence(self._y, _prob))

                    _y = tf.expand_dims(self._y, axis=1)
                    _pi = tf.expand_dims(self._pi[i], axis=0)
                    _mu = tf.expand_dims(_mu, axis=2)
                    _prob = tf.clip_by_value(
                        tf.expand_dims(_prob, axis=1), clip_value_min=1e-6, clip_value_max=1.0)
                    _new_pi = tf.multiply(tf.multiply(_y, _pi), _mu) / _prob
                    _new_pi = tf.reduce_sum(_new_pi, axis=0)
                    _new_pi = keras.activations.softmax(_new_pi)
                    self._pi[i].assign(_new_pi)

                loss /= self._n_estimators

            gradients = model_tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    def predict(self, X):
        res = np.zeros([X.shape[0], self._n_outputs], dtype=np.float32)
        for i in range(self._n_estimators):
            res += tf.matmul(self._call(X, i), self._pi[i])
        return res / self._n_estimators


class LDL_SCL(BaseDeepLDL):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = keras.activations.softmax(self._model(X) + tf.matmul(self._C, self._W))

        kl = tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))

        corr = tf.math.reduce_mean(self._C * keras.losses.mean_squared_error(
            tf.expand_dims(y_pred, 1), tf.expand_dims(self._P, 0)
        ))

        barr = tf.math.reduce_mean(1 / self._C)

        return kl + self._alpha * corr + self._beta * barr

    def fit(self, X, y, n_clusters=5, learning_rate=5e-2, epochs=3000, alpha=1e-3, beta=1e-6):
        super().fit(X, y)

        self._n_clusters = n_clusters
        self._alpha = alpha
        self._beta = beta

        self._P = tf.cast(KMeans(n_clusters=self._n_clusters).fit(self._y).cluster_centers_,
                          dtype=tf.float32)

        self._C = tf.Variable(tf.zeros((self._X.shape[0], self._n_clusters)) + 1e-6,
                              trainable=True)

        self._W = tf.Variable(tf.random.normal((self._n_clusters, self._n_outputs)),
                              trainable=True)

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                        keras.layers.Dense(self._n_outputs, activation=None)])
        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    def predict(self, X):

        C = np.zeros((X.shape[0], self._n_clusters))
        for i in range(self._n_clusters):
            lr = SVR()
            lr.fit(self._X.numpy(), self._C.numpy()[:, i].reshape(-1))
            C[:, i] = lr.predict(X).reshape(1, -1)
        C = tf.cast(C, dtype=tf.float32)

        return keras.activations.softmax(self._model(X) + tf.matmul(C, self._W))


class IncomLDL(BaseDeepLDL):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def _loss(self, X, y):
        y_pred = self._model(X)
        trace_norm = 0.
        for i in self._model.trainable_variables:
            trace_norm += tf.linalg.trace(tf.sqrt(tf.matmul(tf.transpose(i), i)))
        fro_norm = tf.reduce_sum(tf.square(self._mask * (y_pred - y)))
        return fro_norm / 2. + self._alpha * trace_norm

    def fit(self, X, y, mask, alpha=2., learning_rate=5e-2, epochs=5000):
        super().fit(X, y)

        self._alpha = alpha
        self._mask = tf.where(mask, 0., 1.)

        self._model = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_features),
                                        keras.layers.Dense(self._n_outputs, activation='softmax', use_bias=False)])
        self._optimizer = tfa.optimizers.ProximalAdagrad(learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
    def predict(self, X):
        return self._model(X).numpy()
    
    def score(self, X, y, metrics=None):
        if metrics is None:
            metrics = ["chebyshev", "clark", "canberra", "cosine", "intersection"]
        return score(y, self.predict(X), metrics=metrics)


class DeepBFGS():

    @tf.function
    def _assign_new_model_parameters(self, params_1d, model):
        params = tf.dynamic_partition(params_1d, self._part, self._n_tensors)
        for i, (shape, param) in enumerate(zip(self._model_shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    def _get_obj_func(self, model, loss_function, X, y):

        @tf.function
        def _f(params_1d):

            with tf.GradientTape() as tape:
                self._assign_new_model_parameters(params_1d, model)
                loss = loss_function(X, y)

            grads = tape.gradient(loss, model.trainable_variables)
            grads = tf.dynamic_stitch(self._idx, grads)

            return loss, grads

        return _f
    
    def _optimize_bfgs(self, model, loss_function, X, y, max_iterations=50):

        self._model_shapes = tf.shape_n(model.trainable_variables)
        self._n_tensors = len(self._model_shapes)

        count = 0
        self._idx = []
        self._part = []

        for i, shape in enumerate(self._model_shapes):
            n = np.product(shape)
            self._idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            self._part.extend([i]*n)
            count += n

        self._part = tf.constant(self._part)
        
        func = self._get_obj_func(model, loss_function, X, y)
        init_params = tf.dynamic_stitch(self._idx, model.trainable_variables)

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=max_iterations
        )

        self._assign_new_model_parameters(results.position, model)


class LDL_LRR(BaseDeepLDL, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _ranking_loss(self, y_pred, P, W):
        Phat = tf.math.sigmoid((tf.expand_dims(y_pred, -1) - tf.expand_dims(y_pred, 1)) * 100)
        l = ((1 - P) * tf.math.log(tf.clip_by_value(1 - Phat, 1e-9, 1.0)) + \
              P * tf.math.log(tf.clip_by_value(Phat, 1e-9, 1.0))) * W
        return -tf.reduce_sum(l)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)
        kl = tf.math.reduce_mean(keras.losses.kl_divergence(y, y_pred))
        rank = self._ranking_loss(y_pred, self._P, self._W) / (2 * X.shape[0])
        return kl + self._alpha * rank + self._beta * self._l2_reg(self._model)

    def fit(self, X, y, alpha=1e-2, beta=1e-8, max_iterations=50):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta

        P = tf.nn.sigmoid(tf.expand_dims(self._y, -1) - tf.expand_dims(self._y, 1))
        self._P = tf.where(P > .5, 1., 0.)

        self._W = tf.square(tf.expand_dims(self._y, -1) - tf.expand_dims(self._y, 1))

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])

        self._optimize_bfgs(self._model, self._loss, self._X, self._y, max_iterations)

    def predict(self, X):
        return self._model(X).numpy()


class BaseDeepLDLClassifier(BaseDeepLDL):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    def predict_proba(self, X):
        return self._model(X).numpy()
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def score(self, X, y, metrics=None):
        if metrics is None:
            metrics = ["zero_one_loss", "error_probability"]
        return score(y, self.predict_proba(X), metrics=metrics)


class LDL4C(BaseDeepLDLClassifier, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)
        top2 = tf.gather(y_pred, self._top2, axis=1, batch_dims=1)
        margin = tf.reduce_sum(tf.maximum(0., 1. - (top2[:, 0] - top2[:, 1]) / self._rho))
        mae = keras.losses.mean_absolute_error(y, y_pred)
        return tf.reduce_sum(self._entropy * mae) + self._alpha * margin + self._beta * self._l2_reg(self._model)
    
    def fit(self, X, y, max_iterations=50, alpha=1e-2, beta=1e-6, rho=1e-2):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta
        self._rho = rho

        self._top2 = tf.math.top_k(self._y, k=2)[1]
        self._entropy = tf.cast(-tf.reduce_sum(self._y * tf.math.log(self._y) + 1e-7, axis=1), dtype=tf.float32)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])

        self._optimize_bfgs(self._model, self._loss, self._X, self._y, max_iterations=max_iterations)


class LDL_HR(BaseDeepLDLClassifier, DeepBFGS):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)
        
        highest = tf.gather(y_pred, self._highest, axis=1, batch_dims=1)
        rest = tf.gather(y_pred, self._rest, axis=1, batch_dims=1)
        margin = tf.reduce_sum(tf.maximum(0., 1. - (highest - rest) / self._rho))

        real_rest = tf.gather(y, self._rest, axis=1, batch_dims=1)
        rest_mae = tf.reduce_sum(keras.losses.mean_absolute_error(real_rest, rest))

        mae = tf.reduce_sum(keras.losses.mean_absolute_error(self._l, y_pred))

        return mae + self._alpha * margin + self._beta * rest_mae + self._gamma * self._l2_reg(self._model)
    
    def fit(self, X, y, max_iterations=50, alpha=1e-2, beta=1e-2, gamma=1e-6, rho=1e-2):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._rho = rho

        temp = tf.math.top_k(self._y, k=self._n_outputs)[1]
        self._highest = temp[:, 0:1]
        self._rest = temp[:, 1:]

        self._l = tf.one_hot(tf.reshape(self._highest, -1), self._n_outputs)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])

        self._optimize_bfgs(self._model, self._loss, self._X, self._y, max_iterations=max_iterations)


class LDLM(BaseDeepLDLClassifier):

    def __init__(self, n_hidden=None, n_latent=None, random_state=None):
        super().__init__(n_hidden, n_latent, random_state)

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)

        pred_margin = tf.reduce_sum(tf.clip_by_value(
            tf.reduce_sum(tf.abs(self._l - y_pred), axis=1) - self._rho,
            0., float('inf')))
        
        highest = tf.gather(y_pred, self._highest, axis=1, batch_dims=1)
        rest = tf.gather(y_pred, self._rest, axis=1, batch_dims=1)
        label_margin = tf.reduce_sum(tf.maximum(0., 1. - (highest - rest) / self._rho))

        second_margin = tf.reduce_sum(tf.clip_by_value(
            tf.reduce_sum(tf.abs((y - y_pred) * self._neg_l), axis=1) - self._second_margin,
            0., float('inf')))

        return pred_margin + self._alpha * label_margin + \
            self._beta * second_margin + self._gamma * self._l2_reg(self._model)

    def fit(self, X, y, learning_rate=5e-4, epochs=1000,
            alpha=1e-2, beta=1e-2, gamma=1e-6, rho=1e-2):
        super().fit(X, y)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._rho = rho

        temp = tf.math.top_k(self._y, k=self._n_outputs)[1]
        self._highest = temp[:, 0:1]
        self._rest = temp[:, 1:]

        temp = tf.math.top_k(tf.gather(self._y, self._rest, axis=1, batch_dims=1), k=2)[0]
        self._second_margin = temp[:, 0] - temp[:, 1]

        self._l = tf.one_hot(tf.reshape(self._highest, -1), self._n_outputs)
        self._neg_l = tf.where(tf.equal(self._l, 0.), 1., 0.)

        self._model = keras.Sequential(
            [keras.layers.InputLayer(input_shape=self._n_features),
             keras.layers.Dense(self._n_outputs, activation="softmax")])
        
        self._optimizer = keras.optimizers.SGD(learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class _PT(BaseLDL):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = None

    def _preprocessing(self, X, y):
        m, c = y.shape[0], y.shape[1]
        Xr = np.repeat(X, c, axis=0)
        yr = np.tile(np.arange(c), m)
        p = y.reshape(-1) / m

        select = np.random.choice(m*c, size=m*c, p=p)

        return Xr[select], yr[select]

    def fit(self, X, y):
        super().fit(X, y)
        self._X, self._y = self._preprocessing(self._X, self._y)

        self._model.fit(self._X, self._y)

    def predict(self, X):
        return self._model.predict_proba(X)


class PT_Bayes(_PT):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = GaussianNB(var_smoothing=0.1)


class PT_SVM(_PT):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = CalibratedClassifierCV(LinearSVC())


class LDSVR(BaseLDL):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._model = None

    def fit(self, X, y):
        super().fit(X, y)
        self._model = MultiOutputRegressor(SVR(tol=1e-10, gamma=1./(2.*np.mean(pdist(self._X))**2)))

        y = -np.log(1. / (self._y + 1e-7) - 1.)

        self._model.fit(self._X, y)

    def predict(self, X):
        y_pred = 1. / (1. + np.exp(-self._model.predict(X)))
        return y_pred / np.sum(y_pred, axis=1).reshape(-1, 1)


class BaseEnsemble(BaseLDL):

    def __init__(self, estimator=SA_BFGS(), n_estimators=None, random_state=None):
        super().__init__(random_state)
        self._estimator = estimator
        self._n_estimators = n_estimators
        self._estimators = None

    def __len__(self):
        return len(self._estimators)

    def __getitem__(self, index):
        return self._estimators[index]

    def __iter__(self):
        return iter(self._estimators)


class DF_LDL(BaseEnsemble):

    def __init__(self, estimator=SA_BFGS(), random_state=None):
        super().__init__(estimator, None, random_state)

    def fit(self, X, y):
        super().fit(X, y)

        m, c = self._y.shape[0], self._y.shape[1]
        L = {}

        for i in range(c):
            for j in range(i + 1, c):

                ss1 = []
                ss2 = []

                for k in range(m):
                    if self._y[k, i] >= self._y[k, j]:
                        ss1.append(k)
                    else:
                        ss2.append(k)

                l1 = copy.deepcopy(self._estimator)
                l1.fit(self._X[ss1], self._y[ss1])
                L[str(i)+","+str(j)] = copy.deepcopy(l1)

                l2 = copy.deepcopy(self._estimator)
                l2.fit(self._X[ss2], self._y[ss2])
                L[str(j)+","+str(i)] = copy.deepcopy(l2)

        self._estimators = L

        self._knn = AA_KNN()
        self._knn.fit(self._X, self._y)

    def predict(self, X):

        m, c = X.shape[0], self._y.shape[1]
        p_knn = self._knn.predict(X)
        p = np.zeros((m, c), dtype=np.float32)

        for k in range(m):
            for i in range(c):
                for j in range(i + 1, c):

                    if p_knn[k, i] >= p_knn[k, j]:
                        l = self._estimators[str(i)+","+str(j)]
                    else:
                        l = self._estimators[str(j)+","+str(i)]

                    p[k] += l.predict(X[k].reshape(1, -1)).reshape(-1)

        return p / (c * (c - 1) / 2)


class AdaBoostLDL(BaseEnsemble):

    def __init__(self, estimator=SA_BFGS(), n_estimators=10, random_state=None):
        super().__init__(estimator, n_estimators, random_state)

    def fit(self, X, y, loss=sort_loss, alpha=1.):
        super().fit(X, y)

        m = self._X.shape[0]
        p = np.ones((m,)) / m

        self._loss = np.zeros((self._n_estimators, m))
        self._estimators = []
        for i in range(self._n_estimators):
            select = np.random.choice(m, size=m, p=p)
            X_train, y_train = self._X[select], self._y[select]

            model = copy.deepcopy(self._estimator)
            model.fit(X_train, y_train)
            self._estimators.append(copy.deepcopy(model))

            y_pred = model.predict(self._X)
            self._loss[i] = loss(y, y_pred, reduction=None)
            p += alpha * (self._loss[i] / np.sum(self._loss))
            p /= np.sum(p)

    def predict(self, X):
        w = np.sum(self._loss, axis=1)
        w /= np.sum(w)
        y = np.zeros((X.shape[0], self._n_outputs))
        for i in range(self._n_estimators):
            y += w[i] * self._estimators[i].predict(X)
        return y


class SSG_LDL(BaseLDL):

    def __init__(self, n=300, k=5, fx=0.5, fy=0.5, random_state=None):
        super().__init__(random_state)

        self._n = n
        self._k = k
        self._fx = fx
        self._fy = fy

    def _select_sample(self):
        total_dist = np.sum(self._dist)
        r = np.random.rand() * total_dist
        for i in range(self._X.shape[0]):
            r -= self._dist[i]
            if r < 0:
                return i

    def _create_synthetic_sample(self, i):
        _, index = self._knn.kneighbors(np.concatenate([self._X[i], self._y[i]]).reshape(1, -1))
        nnarray = index.reshape(-1)
        nn = np.random.randint(1, self._k)

        dif = self._X[nnarray[nn-1]] - self._X[i]
        gap = np.random.random(self._X[0].shape)

        X = self._X[i] + gap * dif
        y = np.average(self._y[nnarray], axis=0)

        self._new_X = np.concatenate([self._new_X, X.reshape(1, -1)])
        self._new_y = np.concatenate([self._new_y, y.reshape(1, -1)])

    def fit_transform(self, X, y):
        super().fit(X, y)

        self._dist = self._fx * np.sum(np.linalg.norm(np.repeat([self._X], self._X.shape[0], axis=0)
            .transpose(1, 0, 2) - self._X, axis=2), axis=1) / self._X.shape[0] + \
            self._fy * np.sum(np.linalg.norm(np.repeat([self._y], self._y.shape[0], axis=0)
            .transpose(1, 0, 2) - self._y, axis=2), axis=1) / self._y.shape[0]

        self._knn = NearestNeighbors(n_neighbors=self._k)
        self._knn.fit(np.concatenate([self._X, self._y], axis=1))

        self._new_X = self._X
        self._new_y = self._y

        t = self._X.shape[0] * self._n // 100

        for _ in range(t):
            self._create_synthetic_sample(self._select_sample())

        return self._new_X, self._new_y


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

        y_hat = keras.activations.softmax(samples)
        outputs = self._decoder(samples)

        kl = tf.math.reduce_mean(tfp.distributions.kl_divergence(d, std_d), axis=1)
        rec_X = keras.losses.mean_squared_error(X, outputs)
        rec_y = keras.losses.binary_crossentropy(l, y_hat)

        return tf.math.reduce_sum(kl + rec_X + rec_y)

    def fit_transform(self, X, l, learning_rate=1e-5, epochs=3000):
        super().fit_transform(X, l)

        input_shape = self._n_features + self._n_outputs

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._encoder = keras.Sequential([keras.layers.InputLayer(input_shape=input_shape),
                                          keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                          keras.layers.Dense(self._n_outputs*2, activation=None)])

        self._decoder = keras.Sequential([keras.layers.InputLayer(input_shape=self._n_outputs),
                                          keras.layers.Dense(self._n_hidden, activation='sigmoid'),
                                          keras.layers.Dense(self._n_features, activation=None)])

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


__all__ = ["SA_BFGS", "SA_IIS", "AA_KNN", "AA_BP", "PT_Bayes", "PT_SVM",
           "CPNN", "BCPNN", "ACPNN", "LDSVR",
           "LDLF", "LDL_SCL", "LDL_LRR", "CAD", "QFD2", "CJS",
           "DF_LDL", "AdaBoostLDL",
           "LDL4C", "LDL_HR", "LDLM",
           "IncomLDL",
           "SSG_LDL",
           "FCM", "KM", "LP", "ML", "LEVI"]
