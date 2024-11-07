import logging
import functools
from typing import Optional

import keras
import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from pyldl.algorithms.utils import proj, DEFAULT_METRICS


class _Base:
    """Base class for all models in PyLDL.
    """

    def __init__(self, random_state: Optional[int] = None):
        if random_state is not None:
            np.random.seed(random_state)

        self._n_features = None
        self._n_outputs = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the model.
        """
        self._X = X
        self._y = y
        self._n_features = self._X.shape[1]
        if self._y is not None:
            self._n_outputs = self._y.shape[1]
        return self

    def _not_been_fit(self):
        raise ValueError("The model has not yet been fit. "
                         "Try to call 'fit()' first with some training data.")

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            self._not_been_fit()
        return self._n_features

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            self._not_been_fit()
        return self._n_outputs

    def __str__(self) -> str:
        return self.__class__.__name__


class _BaseLDL(_Base):

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def score(self, X: np.ndarray, y: np.ndarray,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        if metrics is None:
            metrics = DEFAULT_METRICS
        from pyldl.metrics import score
        return score(y, self.predict(X), metrics=metrics, return_dict=return_dict)


class _BaseLE(_Base):

    def fit(self, X: np.ndarray, l: np.ndarray):
        super().fit(X, None)
        self._l = l
        self._n_outputs = self._l.shape[1]
        return self

    def transform(self) -> np.ndarray:
        return self._y

    def fit_transform(self, X: np.ndarray, l: np.ndarray, **kwargs):
        return self.fit(X, l, **kwargs).transform()

    def score(self, y: np.ndarray,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        if metrics is None:
            metrics = DEFAULT_METRICS
        from pyldl.metrics import score
        return score(y, self.transform(), metrics=metrics, return_dict=return_dict)


class BaseLDL(_BaseLDL, BaseEstimator):
    """Let :math:`\\mathcal{X} = \\mathbb{R}^{q}` denote the input space and :math:`\\mathcal{Y} = \\lbrace y_i \\rbrace_{i=1}^{l}` denote the label space. 
    The description degree of :math:`y \\in \\mathcal{Y}` to :math:`\\boldsymbol{x} \\in \\mathcal{X}` is denoted by :math:`d_{\\boldsymbol{x}}^{y}`. 
    Then the label distribution of :math:`\\boldsymbol{x}` is defined as :math:`\\boldsymbol{d} = \\lbrace d_{\\boldsymbol{x}}^{y} \\rbrace_{y \\in \\mathcal{Y}}`. 
    Note that :math:`\\boldsymbol{d}` is under the constraints of probability simplex, i.e., :math:`\\boldsymbol{d} \\in \\Delta^{l-1}`, 
    where :math:`\\Delta^{l-1} = \\lbrace \\boldsymbol{d} \\in \\mathbb{R}^{l} \\,|\\, \\boldsymbol{d} \\geq 0,\, \\boldsymbol{d}^{\\text{T}} \\boldsymbol{1} = 1 \\rbrace`. 
    Given a training set of :math:`n` samples :math:`\\mathcal{S} = \\lbrace (\\boldsymbol{x}_i,\, \\boldsymbol{d}_i) \\rbrace_{i=1}^{n}`, 
    the goal of LDL is to learn a conditional probability mass function :math:`p(\\boldsymbol{d} \,|\, \\boldsymbol{x})`.
    """
    pass


class BaseLE(_BaseLE, TransformerMixin, BaseEstimator):
    pass


class Base(_Base):

    def fit(self, X, y, **kwargs):
        if issubclass(self.__class__, BaseIncomLDL):
            mask = kwargs.pop("mask")
            BaseIncomLDL.fit(self, X, y, mask, **kwargs)
        elif issubclass(self.__class__, BaseLDL):
            BaseLDL.fit(self, X, y, **kwargs)
        elif issubclass(self.__class__, BaseLE):
            BaseLE.fit(self, X, y, **kwargs)
        else:
            raise TypeError("The model must be a subclass of BaseLDL or BaseLE.")


class BaseADMM(Base):

    def __init__(self, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.EPS_ABS = 1e-4
        self.EPS_REL = 1e-3
        self.EPS_ERR = 1e-3
        self._olds = {}

    def _update_W(self):
        pass

    def _update_Z(self):
        pass

    def _update_V(self):
        self._V = self._V + self._rho * (self._X @ self._W - self._Z)

    def _before_train(self):
        pass

    def _get_default_model(self):
        _W = np.ones((self._n_features, self._n_outputs))
        _Z = np.ones((self._X.shape[0], self._n_outputs))
        _V = np.ones((self._X.shape[0], self._n_outputs))
        return _W, _Z, _V

    @property
    def constraint(self):
        return [[self._Z, self._X @ self._W]]

    @property
    def params(self):
        return [self._W, self._Z]

    @property
    def Vs(self):
        return [self._V]

    def _primal_residual(self):
        return np.array([np.linalg.norm(c[0] - c[1], 'fro') for c in self.constraint])

    def _dual_residual(self):
        return np.array([np.linalg.norm(self._rho * (c[0] - self._olds[i]), 'fro') for i, c in enumerate(self.constraint)])

    def _primal_eps(self):
        return np.sqrt(self._X.shape[0]) * self.EPS_ABS + self.EPS_REL * np.array([np.maximum(
            np.linalg.norm(c[0], 'fro'), np.linalg.norm(c[1], 'fro')
        ) for c in self.constraint])

    def _dual_eps(self):
        return np.sqrt(self._n_outputs) * self.EPS_ABS + self.EPS_REL * np.array([
            np.linalg.norm(v, 'fro') for v in self.Vs
        ])

    def _err(self):
        err = 0.
        for i, c in enumerate(self.params):
            err = np.maximum(err, np.abs(c - self._olds[i]).max())
        return err

    def _restore(self):
        if self._stopping_criterion == 'primal_dual':
            for i, c in enumerate(self.constraint):
                self._olds[i] = c[0]
        elif self._stopping_criterion == 'error':
            for i, c in enumerate(self.params):
                self._olds[i] = c

    def _converged(self):
        if self._stopping_criterion == 'primal_dual':
            return np.all(self._primal_residual() <= self._primal_eps()) and np.all(self._dual_residual() <= self._dual_eps())
        elif self._stopping_criterion == 'error':
            return self._err() <= self.EPS_ERR
        elif self._stopping_criterion is None:
            return False

    def fit(self, X, y, *, max_iterations=100, rho=1., stopping_criterion='primal_dual', **kwargs):
        super().fit(X, y, **kwargs)
        self._rho = rho
        self._max_iterations = max_iterations
        self._stopping_criterion = stopping_criterion

        self._before_train()

        self._W, self._Z, self._V = self._get_default_model()

        for i in range(self._max_iterations):
            self._current_iteration = i + 1
            self._restore()
            self._update_W()
            self._update_Z()
            self._update_V()
            if self._converged():
                logging.info(f"Converged in {self._current_iteration} iterations.")
                break

        return self

    def predict(self, X):
        return proj(X @ self._W)


class BaseIncomLDL(BaseLDL):

    @staticmethod
    def repair(y, mask):
        b = np.sum(mask, axis=1)
        c = 1 - np.sum(y, axis=1)
        b0 = b.copy()
        b[b == 0] = 1
        d = c / b
        A = d[:, np.newaxis] * mask
        B = (b0 == 1)[:, np.newaxis] * A
        return y + B, (B == 0) & mask

    def fit(self, X, y, mask):
        super().fit(X, y)
        self._y, mask = self.repair(self._y, mask)
        self._mask = np.where(mask, 0., 1.)


class _BaseDeep(keras.Model):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None):
        keras.Model.__init__(self)
        if not random_state is None:
            tf.random.set_seed(random_state)
        self._n_latent = n_latent
        self._n_hidden = n_hidden

    @staticmethod
    @tf.function
    def _l2_reg(model):
        if isinstance(model, keras.Model):
            return tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in model.trainable_variables]) / 2.
        elif isinstance(model, tf.Tensor):
            return tf.reduce_sum(tf.square(model)) / 2.
        else:
            raise TypeError("Input must be a keras.Model or tf.Tensor.")

    @staticmethod
    @tf.function
    def loss_function(y, y_pred):
        return tf.math.reduce_mean(keras.losses.mean_squared_error(y, y_pred))

    def _call(self, X):
        return self._model(X)

    @staticmethod
    def get_2layer_model(n_features, n_outputs, softmax=True):
        activation = 'softmax' if softmax else None
        return keras.Sequential([keras.layers.InputLayer(input_shape=(n_features,)),
                                 keras.layers.Dense(n_outputs, activation=activation, use_bias=False)])

    @staticmethod
    def get_3layer_model(n_features, n_hidden, n_outputs):
        return keras.Sequential([keras.layers.InputLayer(input_shape=(n_features,)),
                                 keras.layers.Dense(n_hidden, activation='sigmoid'),
                                 keras.layers.Dense(n_outputs, activation='softmax')])

    def _get_default_model(self):
        return self.get_3layer_model(self._n_features, self._n_hidden, self._n_outputs)

    def _before_train(self):
        pass

    @tf.function
    def _loss(self, X, y, start, end):
        y_pred = self._call(X)
        return self.loss_function(y, y_pred)

    def fit(self, X, y, model=None, metrics=None, verbose=0):
        self._verbose = verbose
        self._metrics = metrics or []
        self._before_train()
        self._model = model or self._get_default_model()


class BaseDeepLDL(BaseLDL, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None):
        BaseLDL.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit(self, X, y, **kwargs):
        BaseLDL.fit(self, X, y)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._y = tf.cast(self._y, dtype=tf.float32)
        _BaseDeep.fit(self, self._X, self._y, **kwargs)
        return self

    def predict(self, X):
        return self._call(X).numpy()


class BaseDeepLE(BaseLE, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None):
        BaseLE.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit(self, X, l, **kwargs):
        BaseLE.fit(self, X, l)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._l = tf.cast(self._l, dtype=tf.float32)
        _BaseDeep.fit(self, self._X, self._l, **kwargs)
        return self

    def transform(self):
        return keras.activations.softmax(self._call(self._X)).numpy()


class BaseDeepLDLClassifier(BaseDeepLDL):

    def predict_proba(self, X):
        return self._model(X).numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, metrics=None, return_dict=False):
        if metrics is None:
            metrics = ["zero_one_loss", "error_probability"]
        from pyldl.metrics import score
        return score(y, self.predict_proba(X), metrics=metrics, return_dict=return_dict)


class BaseDeep(_BaseDeep):

    def fit(self, X, y, **kwargs):
        if issubclass(self.__class__, BaseDeepLDL):
            BaseDeepLDL.fit(self, X, y, **kwargs)
        elif issubclass(self.__class__, BaseDeepLE):
            BaseDeepLE.fit(self, X, y, **kwargs)
        else:
            raise TypeError("The model must be a subclass of BaseDeepLDL or BaseDeepLE.")


class BaseGD(BaseDeep):

    def _get_default_optimizer(self):
        return keras.optimizers.SGD()

    def train(self, X, y, epochs, batch_size, loss, trainable_variables, callbacks=None, X_val=None, y_val=None):

        data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

        if not isinstance(callbacks, keras.callbacks.CallbackList):
            callbacks = keras.callbacks.CallbackList(callbacks, model=self)
        callbacks.on_train_begin()
        if self._verbose != 0:
            progbar = keras.utils.Progbar(epochs, stateful_metrics=self._metrics + ['loss'])

        self.stop_training = False
        for epoch in range(epochs):
            if self.stop_training:
                break
            callbacks.on_epoch_begin(epoch)

            total_loss = 0.
            for step, batch in enumerate(data):
                start = step * batch_size
                end = min(start + batch_size, X.shape[0])
                callbacks.on_train_batch_begin(step)
                with tf.GradientTape() as tape:
                    l = loss(batch[0], batch[1], start, end)
                    total_loss += l
                gradients = tape.gradient(l, trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, trainable_variables))
                callbacks.on_train_batch_end(step)

            scores = {}
            if y_val is not None:
                if X_val is not None:
                    if issubclass(self.__class__, BaseDeepLDLClassifier):
                        y_val_pred = self.predict_proba(X_val)
                    elif issubclass(self.__class__, BaseDeepLDL):
                        y_val_pred = self.predict(X_val)
                elif issubclass(self.__class__, BaseDeepLE):
                    y_val_pred = self.transform()

                from pyldl.metrics import score
                scores = score(y_val, y_val_pred, metrics=self._metrics, return_dict=True)

            callbacks.on_epoch_end(epoch + 1, {"scores": scores, "loss": total_loss})
            if self._verbose != 0:
                progbar.update(epoch + 1, values=[('loss', total_loss)] + list(scores.items()),
                               finalize=self.stop_training or epochs == epoch + 1)

        callbacks.on_train_end()

    def fit(self, X, y, *, epochs=1000, batch_size=None, optimizer=None,
            X_val=None, y_val=None, callbacks=None, **kwargs):
        super().fit(X, y, **kwargs)

        self._batch_size = batch_size or self._X.shape[0]
        self._optimizer = optimizer or self._get_default_optimizer()
        self.train(self._X, self._y if issubclass(self.__class__, BaseDeepLDL) else self._l,
                   epochs, self._batch_size, self._loss, self.trainable_variables, callbacks, X_val, y_val)

        return self


class BaseAdam(BaseGD):
    def _get_default_optimizer(self):
        return keras.optimizers.Adam()


class BaseBFGS(BaseDeep):

    @staticmethod
    def make_val_and_grad_fn(value_fn):
        @functools.wraps(value_fn)
        def val_and_grad(x):
            return tfp.math.value_and_gradient(value_fn, x)
        return val_and_grad

    @tf.function
    def _params2model(self, params_1d):
        model_variables = []
        params = tf.dynamic_partition(params_1d, self._part, self._n_tensors)
        for (shape, param) in zip(self._model_shapes, params):
            model_variables.append(tf.reshape(param, shape))
        return model_variables

    def _assign_new_model_parameters(self, params_1d):
        for i, j in enumerate(self._params2model(params_1d)):
            self._model.trainable_variables[i].assign(j)

    def _loss(self, params_1d):
        y_pred = keras.activations.softmax(self._X @ self._params2model(params_1d)[0])
        return self.loss_function(self._y if issubclass(self.__class__, BaseDeepLDL) else self._l, y_pred)

    def _get_obj_func(self):
        return self.make_val_and_grad_fn(self._loss)

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _optimize_bfgs(self, max_iterations):

        self._model_shapes = tf.shape_n(self._model.trainable_variables)
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

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=self._get_obj_func(),
            initial_position=tf.dynamic_stitch(self._idx, self._model.trainable_variables),
            max_iterations=max_iterations
        )

        self._assign_new_model_parameters(results.position)

    def fit(self, X, y, *, max_iterations=50, **kwargs):
        super().fit(X, y, **kwargs)
        self._optimize_bfgs(max_iterations)
        return self


class BaseEnsemble(BaseLDL):

    def __init__(self, estimator: BaseLDL, n_estimators: Optional[int] = None,
                 random_state: Optional[int] = None):
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
