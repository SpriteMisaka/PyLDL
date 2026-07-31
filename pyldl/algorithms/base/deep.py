from functools import wraps

import numpy as np

import keras
import keras.ops as ops

from .shallow import _path_suffix, BaseLDL, BaseLE, BaseLDLClassifier


class _BaseDeep(keras.Model):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None, **kwargs):
        keras.Model.__init__(self, **kwargs)
        self._n_hidden = n_hidden
        self._n_latent = n_latent
        if random_state is not None:
            keras.utils.set_random_seed(random_state)
        self._model = None

        backend = keras.backend.backend()
        if backend == 'tensorflow':
            self._predict = lambda X: self._call(X).numpy()
        elif backend == 'torch':
            self._predict = lambda X: self._call(X).detach().cpu().numpy()

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
    def _l2_reg(model):
        if isinstance(model, keras.Model):
            return ops.sum([ops.sum(ops.square(v)) for v in model.trainable_variables]) / 2.
        else:
            return ops.sum(ops.square(model)) / 2.

    @staticmethod
    def loss_function(Y, Y_pred):
        return ops.mean(keras.losses.mean_squared_error(Y, Y_pred))

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
        self._X = ops.cast(self._X, dtype="float32")
        self._D = ops.cast(self._D, dtype="float32")
        _BaseDeep.fit(self, self._X, self._D, **kwargs)
        return self

    def predict(self, X):
        return self._predict(X)


class BaseDeepLE(BaseLE, _BaseDeep):

    def __init__(self, n_hidden=64, n_latent=None, random_state=None, **kwargs):
        BaseLE.__init__(self, random_state=random_state)
        _BaseDeep.__init__(self, n_hidden, n_latent, random_state=random_state, **kwargs)

    def fit(self, X, L, **kwargs):
        BaseLE.fit(self, X, L)
        self._X = ops.cast(self._X, dtype="float32")
        self._L = ops.cast(self._L, dtype="float32")
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_backend()

    def _setup_backend(self):
        backend = keras.backend.backend()
        if backend == 'tensorflow':
            self._train_step_impl = self._tf_train_step
            self._make_dataset = self._tf_make_dataset
        elif backend == 'torch':
            self._train_step_impl = self._torch_train_step
            self._make_dataset = self._torch_make_dataset

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

    def _tf_train_step(self, pair, batch, loss, start, end):
        import tensorflow as tf
        trainable_variables, optimizer = pair
        X, Y = batch
        with tf.GradientTape() as tape:
            l = loss(X, Y, start, end)
            self.total_loss += l
        gradients = tape.gradient(l, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    def _torch_train_step(self, pair, batch, loss, start, end):
        trainable_variables, optimizer = pair
        X, Y = batch
        for var in trainable_variables:
            var.value.grad = None
        l = loss(X, Y, start, end)
        self.total_loss += float(l)
        l.backward()
        grads = [var.value.grad for var in trainable_variables]
        optimizer.apply_gradients(zip(grads, trainable_variables))

    def train_step(self, pair, batch, loss, start, end):
        return self._train_step_impl(pair, batch, loss, start, end)

    def _tf_make_dataset(self, X, Y):
        import tensorflow as tf
        return tf.data.Dataset.from_tensor_slices((X, Y)).batch(self._batch_size)

    def _torch_make_dataset(self, X, Y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(Y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

    def train(self, X, Y, epochs, callbacks, X_val, D_val, L_val):

        data = self._make_dataset(X, Y)

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
                self.train_step((self.trainable_variables, self._optimizer), batch, self._loss, start, end)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_bfgs_backend()

    def _setup_bfgs_backend(self):
        backend = keras.backend.backend()
        if backend == 'tensorflow':
            self._optimize_bfgs = self._tf_optimize_bfgs
        elif backend == 'torch':
            self._optimize_bfgs = self._torch_optimize_bfgs

    @staticmethod
    def loss_function(Y, Y_pred):
        return ops.mean(keras.losses.kl_divergence(Y, Y_pred))

    def _loss(self, params_1d):
        pred = keras.activations.softmax(self._X @ self._params2model(params_1d)[0])
        return self.loss_function(self._D if issubclass(self.__class__, BaseDeepLDL) else self._L, pred)

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _tf_optimize_bfgs(self, max_iterations):
        import tensorflow as tf
        import tensorflow_probability as tfp

        self._model_shapes_np = [tuple(v.shape) for v in self._model.trainable_variables]

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

        @wraps(self._loss)
        def obj_func(x):
            return tfp.math.value_and_gradient(self._loss, x)

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=obj_func,
            initial_position=tf.dynamic_stitch(self._idx, self._model.trainable_variables),
            max_iterations=max_iterations
        )

        self._assign_new_model_parameters(results.position)

    def _torch_optimize_bfgs(self, max_iterations):
        import torch
        import numpy as np

        self._model_shapes_np = [tuple(v.shape) for v in self._model.trainable_variables]

        flat_parts = [v.value.detach().flatten() for v in self._model.trainable_variables]
        param = torch.nn.Parameter(torch.cat(flat_parts))

        optimizer = torch.optim.LBFGS(
            [param],
            max_iter=max_iterations,
            line_search_fn='strong_wolfe',
        )

        def closure():
            optimizer.zero_grad()
            loss = self._loss(param)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            offset = 0
            for v in self._model.trainable_variables:
                n = int(np.prod(v.shape))
                v.assign(param.data[offset:offset+n].reshape(v.shape))
                offset += n

    def _params2model(self, params_1d):
        params = []
        offset = 0
        for shape in self._model_shapes_np:
            n = int(np.prod(shape))
            params.append(ops.reshape(params_1d[offset:offset+n], shape))
            offset += n
        return params

    def _assign_new_model_parameters(self, params_1d):
        for i, j in enumerate(self._params2model(params_1d)):
            self._model.trainable_variables[i].assign(j)

    def fit(self, X, Y, *, max_iterations=50, **kwargs):
        super().fit(X, Y, **kwargs)
        self._optimize_bfgs(max_iterations)
        return self
