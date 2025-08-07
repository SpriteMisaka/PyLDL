import pickle
import logging

from functools import wraps
from pathlib import Path
from typing import Optional

import keras
import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np
from sklearn.base import BaseEstimator

from pyldl.algorithms.utils import proj, DEFAULT_METRICS


def _path_suffix(suffix):
    def decorator(func):
        @wraps(func)
        def wrapper(self, path: str, *args, **kwargs):
            file = Path(path).with_suffix(suffix)
            return func(self, file, *args, **kwargs)
        return wrapper
    return decorator


class _Base:
    """Base class for all models in PyLDL.
    """

    def __init__(self, random_state: Optional[int] = None):
        if random_state is not None:
            np.random.seed(random_state)
        self._n_features = None
        self._n_outputs = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the model.
        """
        self._X = X
        self._n_samples = self._X.shape[0]
        self._n_features = self._X.shape[1]
        self._n_outputs = Y.shape[1]
        return self

    @_path_suffix(".pkl")
    def dump(self, file: str):
        """Save the model to a file.
        """
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    @_path_suffix(".pkl")
    def load(cls, file: str):
        """Load the model from a file.
        """
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, cls):
                return obj
            new_obj = cls.__new__(cls)
            new_obj.__dict__.update(obj.__dict__)
            return new_obj

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

    def predict(self) -> np.ndarray:
        raise NotImplementedError("The 'predict()' method is not implemented.")

    def fit(self, X: np.ndarray, D: np.ndarray):
        super().fit(X, D)
        self._D = D
        return self

    def score(self, X: np.ndarray, D: np.ndarray,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        if metrics is None:
            metrics = DEFAULT_METRICS
        from pyldl.metrics import score
        return score(D, self.predict(X), metrics=metrics, return_dict=return_dict)


class _BaseLE(_Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._D = None

    def fit(self, X: np.ndarray, L: np.ndarray):
        super().fit(X, L)
        self._L = L
        return self

    def transform(self, *args, **kwargs) -> np.ndarray:
        return self._D

    def fit_transform(self, X: np.ndarray, L: np.ndarray, **kwargs):
        return self.fit(X, L, **kwargs).transform(X, L)

    def score(self, D: np.ndarray, X: np.ndarray = None, L: np.ndarray = None,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        if metrics is None:
            metrics = DEFAULT_METRICS
        from pyldl.metrics import score
        return score(D, self.transform(X, L), metrics=metrics, return_dict=return_dict)


class BaseLDL(_BaseLDL, BaseEstimator):
    """Base class for all LDL models in PyLDL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseLE(_BaseLE, BaseEstimator):
    """Base class for all LE models in PyLDL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Base(_Base):

    def fit(self, X, Y, **kwargs):
        if isinstance(self, BaseIncomLDL):
            mask = kwargs.pop("mask")
            BaseIncomLDL.fit(self, X, Y, mask, **kwargs)
        elif isinstance(self, BaseLDL):
            BaseLDL.fit(self, X, Y, **kwargs)
        elif isinstance(self, BaseLE):
            BaseLE.fit(self, X, Y, **kwargs)
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
        assert self._n_features is not None and self._n_outputs is not None
        _W = np.ones((self._n_features, self._n_outputs))
        _Z = np.ones((self._n_samples, self._n_outputs))
        _V = np.ones((self._n_samples, self._n_outputs))
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
        return np.sqrt(self._n_samples) * self.EPS_ABS + self.EPS_REL * np.array([np.maximum(
            np.linalg.norm(c[0], 'fro'), np.linalg.norm(c[1], 'fro')
        ) for c in self.constraint])

    def _dual_eps(self):
        assert self._n_outputs is not None
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

    def fit(self, X, Y, *, max_iterations=100, rho=1., stopping_criterion='primal_dual', **kwargs):
        super().fit(X, Y, **kwargs)
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
    """Base class for all IncomLDL models in PyLDL.
    """

    @staticmethod
    def repair(D, mask):
        b = np.sum(mask, axis=1)
        c = 1 - np.sum(D, axis=1)
        b0 = b.copy()
        b[b == 0] = 1
        d = c / b
        A = d[:, np.newaxis] * mask
        B = (b0 == 1)[:, np.newaxis] * A
        return D + B, (B == 0) & mask

    def fit(self, X, D, mask):
        super().fit(X, D)
        self._D, mask = self.repair(self._D, mask)
        self._mask = np.where(mask, 0., 1.)


class _BaseDeep(keras.Model):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None):
        keras.Model.__init__(self)
        if random_state is not None:
            tf.random.set_seed(random_state)
        self._n_latent = n_latent
        self._n_hidden = n_hidden

    @staticmethod
    @tf.function
    def _l2_reg(model):
        if isinstance(model, keras.Model):
            return tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in model.trainable_variables]) / 2.
        else:
            return tf.reduce_sum(tf.square(model)) / 2.

    @staticmethod
    @tf.function
    def loss_function(Y, Y_pred):
        return tf.math.reduce_mean(keras.losses.mean_squared_error(Y, Y_pred))

    def _call(self, X):
        return self._model(X)

    @staticmethod
    def get_2layer_model(n_features, n_outputs, activation='softmax'):
        return keras.Sequential([keras.layers.InputLayer(shape=(n_features,)),
                                 keras.layers.Dense(n_outputs, activation=activation, use_bias=False)])

    @staticmethod
    def get_3layer_model(n_features, n_hidden, n_outputs,
                         hidden_activation='sigmoid', output_activation='softmax'):
        return keras.Sequential([keras.layers.InputLayer(shape=(n_features,)),
                                 keras.layers.Dense(n_hidden, activation=hidden_activation),
                                 keras.layers.Dense(n_outputs, activation=output_activation)])

    def _get_default_model(self):
        return self.get_3layer_model(self._n_features, self._n_hidden, self._n_outputs)

    def _before_train(self):
        pass

    @tf.function
    def _loss(self, X, Y, start, end):
        Y_pred = self._call(X)
        return self.loss_function(Y, Y_pred)

    def fit(self, X, Y, model=None, metrics=None, verbose=0):
        self._verbose = verbose
        self._metrics = metrics or []
        self._before_train()
        self._model = model or self._get_default_model()


class BaseDeepLDL(BaseLDL, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None):
        BaseLDL.__init__(self, random_state=random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state=random_state)

    def fit(self, X, D, **kwargs):
        BaseLDL.fit(self, X, D)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._D = tf.cast(self._D, dtype=tf.float32)
        _BaseDeep.fit(self, self._X, self._D, **kwargs)
        return self

    def predict(self, X):
        return self._call(X).numpy()


class BaseDeepLE(BaseLE, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None):
        BaseLE.__init__(self, random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state)

    def fit(self, X, L, **kwargs):
        BaseLE.fit(self, X, L)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._L = tf.cast(self._L, dtype=tf.float32)
        _BaseDeep.fit(self, self._X, self._L, **kwargs)
        return self

    def transform(self, X=None, L=None):
        X = self._X if X is None else X
        return keras.activations.softmax(self._call(self._X)).numpy()


class BaseLDLClassifier(BaseLDL):
    """Base class for all LDL4C models in PyLDL.
    """

    def predict_proba(self):
        raise NotImplementedError("The 'predict_proba()' method is not implemented.")

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, D, metrics=None, return_dict=False):
        if metrics is None:
            metrics = ["zero_one_loss", "error_probability"]
        from pyldl.metrics import score
        return score(D, self.predict_proba(X), metrics=metrics, return_dict=return_dict)


class BaseDeepLDLClassifier(BaseLDLClassifier, BaseDeepLDL):

    def predict_proba(self, X):
        return self._call(X).numpy()


class BaseDeep(_BaseDeep):

    @staticmethod
    def _inspect_dims(h5_path):
        import h5py
        with h5py.File(h5_path, 'r') as f:
            shapes = []
            def _func(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                parse = name.split('/')
                p1 = parse[1]
                n = int(p1.split('_')[1]) if '_' in p1 else 0
                m = -1
                if p1.startswith('sequential'):
                    p3 = parse[3]
                    m = int(p3.split('_')[1]) if '_' in p3 else 0
                elif p1.startswith('dense'):
                    m = 0
                if m >= 0 and len(obj.shape) != 1:
                    shapes.append((n, m, obj.shape))

            f.visititems(_func)
            shapes.sort(key=lambda x: (x[0], x[1]))
            shapes = [i[2] for i in shapes]
            return shapes[0][0], shapes[-1][-1]

    def fit(self, X, Y, **kwargs):
        if issubclass(self.__class__, BaseDeepLDL):
            BaseDeepLDL.fit(self, X, Y, **kwargs)
        elif issubclass(self.__class__, BaseDeepLE):
            BaseDeepLE.fit(self, X, Y, **kwargs)
        else:
            raise TypeError("The model must be a subclass of BaseDeepLDL or BaseDeepLE.")

    @_path_suffix(".weights.h5")
    def dump(self, file: str):
        """Save the model to a file.
        """
        from pyldl.algorithms._algorithm_adaptation import CPNN, LDLF
        if issubclass(self.__class__, CPNN) or issubclass(self.__class__, LDLF):
            self._useless = BaseDeep.get_2layer_model(1, self._n_outputs)
        self.built = True
        self.save_weights(file)

    @classmethod
    @_path_suffix(".weights.h5")
    def load(cls, file: str, **kwargs):
        """Load the model from a file.
        """
        X = kwargs.pop("X", None)
        D = kwargs.pop("D", None)
        if X is None or D is None:
            from pyldl.algorithms._label_enhancement import LEVI
            from pyldl.algorithms._algorithm_adaptation import CPNN, BCPNN, ACPNN
            n_features, n_outputs = cls._inspect_dims(file)
            if cls is LEVI or cls is BCPNN or cls is ACPNN:
                n_features -= n_outputs
            elif cls is CPNN:
                n_features -= 1
            X = np.zeros((1, n_features)) if X is None else X
            D = np.zeros((1, n_outputs)) if D is None else D
        obj = cls()
        BaseDeep.fit(obj, X, D, **kwargs)
        obj.built = True
        obj.load_weights(file)
        return obj


class BaseGD(BaseDeep):

    def _get_default_optimizer(self):
        return keras.optimizers.SGD(1e-2)

    def _calculate_validation_scores(self, X_val, D_val, L_val):
        val = None
        if D_val is not None:
            val = D_val
            if X_val is not None:
                if issubclass(self.__class__, BaseDeepLDLClassifier):
                    val_pred = self.predict_proba(X_val)
                elif issubclass(self.__class__, BaseDeepLDL):
                    val_pred = self.predict(X_val)
        if L_val is not None:
            val = L_val
            if issubclass(self.__class__, BaseDeepLE):
                val_pred = self.transform(X_val, L_val)

        if val is not None:
            from pyldl.metrics import score
            return score(val, val_pred, metrics=self._metrics, return_dict=True)
        return {}

    def train_step(self, batch, loss, trainable_variables, epoch, epochs, start, end):
        with tf.GradientTape() as tape:
            l = loss(batch[0], batch[1], start, end)
            self.total_loss += l
        gradients = tape.gradient(l, trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, trainable_variables))

    def train(self, X, Y, epochs, batch_size, loss, trainable_variables, callbacks=None, X_val=None, D_val=None, L_val=None):

        data = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

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

            self.total_loss = 0.
            for step, batch in enumerate(data):
                start = step * batch_size
                end = min(start + batch_size, X.shape[0])
                callbacks.on_train_batch_begin(step)
                self.train_step(batch, loss, trainable_variables, epoch, epochs, start, end)
                callbacks.on_train_batch_end(step)

            scores = self._calculate_validation_scores(X_val, D_val, L_val)

            callbacks.on_epoch_end(epoch + 1, {"scores": scores, "loss": self.total_loss})
            if self._verbose != 0:
                progbar.update(epoch + 1, values=[('loss', self.total_loss)] + list(scores.items()),
                               finalize=self.stop_training or epochs == epoch + 1)

        callbacks.on_train_end()

    def fit(self, X, Y, *, epochs=1000, batch_size=None, optimizer=None,
            X_val=None, D_val=None, L_val=None, callbacks=None, **kwargs):
        super().fit(X, Y, **kwargs)

        self._batch_size = batch_size or self._n_samples
        self._optimizer = optimizer or self._get_default_optimizer()
        self.train(self._X, self._D if issubclass(self.__class__, BaseDeepLDL) else self._L,
                   epochs, self._batch_size, self._loss, self.trainable_variables, callbacks, X_val, D_val, L_val)

        return self


class BaseAdam(BaseGD):
    def _get_default_optimizer(self):
        return keras.optimizers.Adam(1e-2)


class BaseBFGS(BaseDeep):

    @staticmethod
    def make_val_and_grad_fn(value_fn):
        @wraps(value_fn)
        def val_and_grad(x):
            return tfp.math.value_and_gradient(value_fn, x)
        return val_and_grad

    @staticmethod
    @tf.function
    def loss_function(Y, Y_pred):
        return tf.math.reduce_mean(keras.losses.kl_divergence(Y, Y_pred))

    @tf.function
    def _params2model(self, params_1d):
        params = tf.dynamic_partition(params_1d, self._part, self._n_tensors)
        return [
            tf.reshape(param, shape)
            for shape, param in zip(self._model_shapes, params)
        ]

    def _assign_new_model_parameters(self, params_1d):
        for i, j in enumerate(self._params2model(params_1d)):
            self._model.trainable_variables[i].assign(j)

    def _loss(self, params_1d):
        pred = keras.activations.softmax(self._X @ self._params2model(params_1d)[0])
        return self.loss_function(self._D if issubclass(self.__class__, BaseDeepLDL) else self._L, pred)

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
            n = np.prod(shape)
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

    def fit(self, X, Y, *, max_iterations=50, **kwargs):
        super().fit(X, Y, **kwargs)
        self._optimize_bfgs(max_iterations)
        return self


class BaseEnsemble(BaseLDL):

    def __init__(self, estimator: BaseLDL, n_estimators: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = estimator
        self._n_estimators = n_estimators
        self._estimators = None

    def __len__(self):
        return len(self._estimators)

    def __getitem__(self, index):
        return self._estimators[index]

    def __iter__(self):
        return iter(self._estimators)
