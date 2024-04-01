import keras
import tensorflow as tf
import tensorflow_addons as tfa

from pyldl.metrics import score
from pyldl.algorithms.base import BaseDeepLDL


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
