import numpy as np

import keras


class LDLEarlyStopping(keras.callbacks.Callback):

    def __init__(self, monitor='kl_divergence', patience=None, k=8):
        super().__init__()
        self._monitor = monitor
        self._patience = patience
        self._k = k

    def on_train_begin(self, *_):
        self._wait = 0
        self._stopped_epoch = 0
        self._best_weights = None
        self._wave = False
        if self._monitor == 'loss':
            self._smaller = True
        else:
            from pyldl.metrics import THE_SMALLER_THE_BETTER
            self._smaller = self._monitor in THE_SMALLER_THE_BETTER
            if self._monitor not in self.model._metrics:
                self.model._metrics.append(self._monitor)
            self._prev_pred = self.model.predict(self.model._X)
            self._pc = []
        self._best = np.inf if self._smaller else -np.inf

    def _wave_value(self):
        cur_pred = self.model.predict(self.model._X)
        self._pc.append(np.mean(np.abs(cur_pred - self._prev_pred)))
        self._prev_pred = cur_pred
        return np.mean(self._pc[-self._k:])

    def on_epoch_end(self, epoch, logs=None):
        if self._monitor == 'loss':
            current = logs.get('loss')
        else:
            current = logs.get(self._monitor)
            if current is None:
                current = self._wave_value()
                self._wave = True
        if self._wave:
            condition = current < self._best
        else:
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

    def on_train_end(self, *_):
        if self._patience is None:
            self.model.set_weights(self._best_weights)
        if self.model._verbose != 0 and self._stopped_epoch > 0:
            import logging
            if self._wave:
                logging.info(f"Epoch {self._stopped_epoch}: early stopping "
                             f"(best averaged prediction changes: {self._best}).")
            else:
                logging.info(f"Epoch {self._stopped_epoch}: early stopping "
                             f"(best {self._monitor}: {self._best}).")
