import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

import keras
import tensorflow as tf
from keras import backend as K

from pyldl.algorithms.base import BaseLDL, BaseDeepLDL


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

    def fit(self, X, y, learning_rate=5e-3, epochs=3000, batch_size=32,
            model=None, activation='sigmoid', optimizer='SGD', X_test=None, y_test=None):
        super().fit(X, y)

        self._batch_size = batch_size

        if self._n_hidden is None:
            self._n_hidden = self._n_features * 3 // 2

        self._model = model
        if self._model is None:
            self._model = keras.Sequential([keras.layers.InputLayer(input_shape=(self._n_features,)),
                                        keras.layers.Dense(self._n_hidden, activation=activation),
                                        keras.layers.Dense(self._n_outputs, activation='softmax')])
        self._optimizer = eval(f'keras.optimizers.{optimizer}({learning_rate})')
        data = tf.data.Dataset.from_tensor_slices((self._X, self._y)).batch(self._batch_size)

        for _ in range(epochs):
            total_loss = 0.
            for batch in data:
                with tf.GradientTape() as tape:
                    y_pred = self._model(batch[0])
                    loss = self._loss_function(batch[1], y_pred)
                gradients = tape.gradient(loss, self.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                total_loss += loss

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


class RProp(keras.optimizers.Optimizer):
    
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp, self).__init__(name='rprop', **kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, gradients):
        grads = gradients
        shapes = [K.int_shape(p) for p in params]
        alphas = [K.variable(np.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        self.updates = []

        for param, grad, old_grad, prev_weight_delta, alpha in zip(params, grads,
                                                                   old_grads, prev_weight_deltas,
                                                                   alphas):

            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0), K.maximum(alpha * self.scale_down, self.min_alpha), alpha)
            )

            new_delta = K.switch(K.greater(grad, 0),
                                 -new_alpha,
                                 K.switch(K.less(grad, 0),
                                          new_alpha,
                                          K.zeros_like(new_alpha)))

            weight_delta = K.switch(K.less(grad*old_grad, 0), -prev_weight_delta, new_delta)

            new_param = param + weight_delta

            grad = K.switch(K.less(grad*old_grad, 0), K.zeros_like(grad), grad)

            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(prev_weight_delta, weight_delta))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
