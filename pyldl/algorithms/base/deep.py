from functools import wraps

import numpy as np

import keras
import tensorflow as tf

import tensorflow_probability as tfp

from .shallow import _path_suffix, BaseLDL, BaseLE, BaseLDLClassifier


class _BaseDeep(keras.Model):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None, **kwargs):
        keras.Model.__init__(self, **kwargs)
        self._n_hidden = n_hidden
        self._n_latent = n_latent
        if random_state is not None:
            tf.random.set_seed(random_state)
        self._model = None

    _serialize_objects = ['_model']

    def get_config(self):
        config = super().get_config()
        for i in self._serialize_objects:
            config[i] = keras.saving.serialize_keras_object(getattr(self, i))
        return config

    @classmethod
    def from_config(cls, config):
        s = {}
        for i in cls._serialize_objects:
            model_config = config.pop(i)
            s[i] = keras.saving.deserialize_keras_object(model_config)
        obj = cls(**config)
        for i in cls._serialize_objects:
             setattr(obj, i, s[i])
        return obj

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
        return keras.Sequential([keras.layers.InputLayer((n_features,)),
                                 keras.layers.Dense(n_outputs, activation=activation, use_bias=False)])

    @staticmethod
    def get_3layer_model(n_features, n_hidden, n_outputs,
                         hidden_activation='sigmoid', output_activation='softmax'):
        return keras.Sequential([keras.layers.InputLayer((n_features,)),
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

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def n_latent(self):
        return self._n_latent


class BaseDeepLDL(BaseLDL, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None, **kwargs):
        BaseLDL.__init__(self, random_state=random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state=random_state, **kwargs)

    def fit(self, X, D, **kwargs):
        BaseLDL.fit(self, X, D)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._D = tf.cast(self._D, dtype=tf.float32)
        _BaseDeep.fit(self, self._X, self._D, **kwargs)
        return self

    def predict(self, X):
        return self._call(X).numpy()


class BaseDeepLE(BaseLE, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None, **kwargs):
        BaseLE.__init__(self, random_state=random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state=random_state, **kwargs)

    def fit(self, X, L, **kwargs):
        BaseLE.fit(self, X, L)
        self._X = tf.cast(self._X, dtype=tf.float32)
        self._L = tf.cast(self._L, dtype=tf.float32)
        _BaseDeep.fit(self, self._X, self._L, **kwargs)
        return self

    def transform(self, X=None, L=None):
        X = self._X if X is None else X
        return keras.activations.softmax(self._call(self._X)).numpy()


class BaseDeepLDLClassifier(BaseLDLClassifier, BaseDeepLDL):

    def predict_proba(self, X):
        return self._call(X).numpy()


class BaseDeep(_BaseDeep):

    def fit(self, X, Y, **kwargs):
        if issubclass(self.__class__, BaseDeepLDL):
            BaseDeepLDL.fit(self, X, Y, **kwargs)
        elif issubclass(self.__class__, BaseDeepLE):
            BaseDeepLE.fit(self, X, Y, **kwargs)
        else:
            raise TypeError("The model must be a subclass of BaseDeepLDL or BaseDeepLE.")
        self.built = True

    @_path_suffix(".keras")
    def dump(self, file: str):
        """Save the model to a file.
        """
        self.save(file)

    @classmethod
    @_path_suffix(".keras")
    def load(cls, file: str, **kwargs):
        """Load the model from a file.
        """
        return keras.models.load_model(file, custom_objects=kwargs)


class BaseGD(BaseDeep):

    def _get_default_optimizer(self):
        return keras.optimizers.SGD(1e-3)

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

    def train_step(self, batch, loss, trainable_variables, optimizer, epoch, epochs, start, end):
        with tf.GradientTape() as tape:
            l = loss(batch[0], batch[1], start, end)
            self.total_loss += l
        gradients = tape.gradient(l, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    def train(self, X, Y, epochs, callbacks, X_val, D_val, L_val):

        data = tf.data.Dataset.from_tensor_slices((X, Y)).batch(self._batch_size)

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
                start = step * self._batch_size
                end = min(start + self._batch_size, X.shape[0])
                callbacks.on_train_batch_begin(step)
                self.train_step(batch, self._loss, self.trainable_variables, self._optimizer, epoch, epochs, start, end)
                callbacks.on_train_batch_end(step)

            scores = self._calculate_validation_scores(X_val, D_val, L_val)

            callbacks.on_epoch_end(epoch + 1, {**scores, "loss": self.total_loss})
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
                   epochs, callbacks, X_val, D_val, L_val)

        return self


class BaseAdam(BaseGD):
    def _get_default_optimizer(self):
        return keras.optimizers.Adam(1e-3)


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
