import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import keras
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import tensorflow_probability as tfp

from pyldl.metrics import score


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

    def __str__(self):
        return self.__class__.__name__


class _BaseLDL(_Base):

    def predict(self, X):
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
        keras.Model.__init__(self)
        if not random_state is None:
            tf.random.set_seed(random_state)
        self._n_latent = n_latent
        self._n_hidden = n_hidden

    @staticmethod
    @tf.function
    def _l2_reg(model):
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
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=max_iterations
        )

        self._assign_new_model_parameters(results.position, model)


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


class BaseEnsemble(BaseLDL):

    def __init__(self, estimator, n_estimators=None, random_state=None):
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
