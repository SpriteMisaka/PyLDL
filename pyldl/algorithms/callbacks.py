import numpy as np

import keras
import tensorflow as tf


class LDLEarlyStopping(keras.callbacks.Callback):

    def __init__(self, monitor='kl_divergence', patience=None):
        super().__init__()
        self._monitor = monitor
        self._patience = patience

    def on_train_begin(self, logs=None):
        self._wait = 0
        self._stopped_epoch = 0
        if self._monitor == 'loss':
            self._smaller = True
        else:
            from pyldl.algorithms.utils import THE_SMALLER_THE_BETTER
            self._smaller = self._monitor in THE_SMALLER_THE_BETTER
            if self._monitor not in self.model._metrics:
                self.model._metrics.append(self._monitor)
        self._best = np.inf if self._smaller else 0.
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss") if self._monitor == 'loss' else logs.get(self._monitor)
        condition = np.less(current, self._best)
        if not self._smaller:
            condition = not condition
        if condition:
            self._best = current
            self._wait = 0
            self._best_weights = self.model.get_weights()
        else:
            self._wait += 1
            if self._patience is not None and self._wait >= self._patience:
                self._stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self._best_weights)

    def on_train_end(self, logs=None):
        if self._patience is None:
            self.model.set_weights(self._best_weights)
        if self.model._verbose != 0 and self._stopped_epoch > 0:
            tf.print(f"Epoch {self._stopped_epoch}: early stopping (best {self._monitor}: {self._best}).")
