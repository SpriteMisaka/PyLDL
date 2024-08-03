import os
import logging
import requests

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from pyldl.algorithms.base import BaseLDL, BaseLE
from pyldl.metrics import THE_SMALLER_THE_BETTER


class LDLEarlyStopping(keras.callbacks.Callback):

    def __init__(self, monitor='kl_divergence', patience=10):
        super().__init__()
        self._monitor = monitor
        self._patience = patience

    def on_train_begin(self, logs=None):
        self._wait = 0
        self._stopped_epoch = 0
        if self._monitor == 'loss':
            self._smaller = True
        else:
            self._smaller = self._monitor in THE_SMALLER_THE_BETTER
            if self._monitor not in self.model._metrics:
                self.model._metrics.append(self._monitor)
        self._best = np.Inf if self._smaller else 0.
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss") if self._monitor == 'loss' else logs.get("scores").get(self._monitor)
        condition = np.less(current, self._best)
        if not self._smaller:
            condition = not condition
        if condition:
            self._best = current
            self._wait = 0
            self._best_weights = self.model.get_weights()
        else:
            self._wait += 1
            if self._wait >= self._patience:
                self._stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self._best_weights)

    def on_train_end(self, logs=None):
        if self.model._verbose != 0 and self._stopped_epoch > 0:
            tf.print(f"Epoch {self._stopped_epoch}: early stopping (best {self._monitor}: {self._best}).")


def load_dataset(name, dir='dataset'):
    if not os.path.exists(dir):
        logging.info(f'Directory {dir} does not exist, creating it.')
        os.makedirs(dir)
    dataset_path = os.path.join(dir, name+'.mat')
    if not os.path.exists(dataset_path):
        logging.info(f'Dataset {name}.mat does not exist, downloading it now, please wait...')
        url = f'https://raw.githubusercontent.com/SpriteMisaka/PyLDL/main/dataset/{name}.mat'
        response = requests.get(url)
        if response.status_code == 200:
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            logging.info(f'Dataset {name}.mat downloaded successfully.')
        else:
            raise ValueError(f'Failed to download {name}.mat')
    data = sio.loadmat(dataset_path)
    return data['features'], data['labels']


def random_missing(y, missing_rate=.9):
    if missing_rate <= 0. or missing_rate >= 1.:
        raise ValueError("Invalid missing rate, which should be in the range (0, 1).")
    missing_mask = np.random.rand(*y.shape) < missing_rate
    missing_y = y.copy()
    missing_y[missing_mask] = np.nan
    missing_y[np.isnan(missing_y)] = 0.
    return missing_y, missing_mask


def binaryzation(y, method='threshold', param=None):
    r = np.argsort(np.argsort(y))

    if method == 'threshold':
        if param is None:
            param = .5
        elif not isinstance(param, float) or param < 0. or param >= 1.:
            raise ValueError("Invalid param, when method is 'threshold', "
                             "param should be a float in the range [0, 1).")
        b = np.sort(y.T, axis=0)[::-1]
        cs = np.cumsum(b, axis=0)
        m = np.argmax(cs >= param, axis=0)
        return np.where(r >= y.shape[1] - m.reshape(-1, 1) - 1, 1, 0)

    elif method == 'topk':
        if param is None:
            param = y.shape[1] // 2
        elif not isinstance(param, int) or param < 1 or param >= y.shape[1]:
            raise ValueError("Invalid param, when method is 'topk', "
                             "param should be an integer in the range [1, number_of_labels).")
        return np.where(r >= y.shape[1] - param, 1, 0)

    else:
        raise ValueError("Invalid method, which should be 'threshold' or 'topk'.")


def artificial(X, a=1., b=.5, c=.2, d=1.,
               w1=np.array([[4., 2., 1.]]),
               w2=np.array([[1., 2., 4.]]),
               w3=np.array([[1., 4., 2.]]),
               lambda1=.01, lambda2=.01):
    t = a * X + b * X**2 + c * X**3 + d
    psi1 = np.matmul(t, w1.T)**2
    psi2 = (np.matmul(t, w2.T) + lambda1 * psi1)**2
    psi3 = (np.matmul(t, w3.T) + lambda2 * psi2)**2
    y = np.concatenate([psi1, psi2, psi3], axis=1)
    return y / np.sum(y, axis=1).reshape(-1, 1)


def make_ldl(n_samples=200, **kwargs):
    X = np.random.uniform(-1, 1, (n_samples, 3))
    y = artificial(X, **kwargs)
    return X, y


def plot_artificial(n_samples=50, model=None, file_name=None, **kwargs):

    x1 = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    x2 = np.linspace(-1, 1, n_samples).reshape(-1, 1)

    a1, a2 = np.meshgrid(x1, x2)
    a3 = np.sin((a1 + a2) * np.pi)
    aa = np.concatenate([np.expand_dims(a1, axis=2),
                         np.expand_dims(a2, axis=2)], axis=2)
    bb = aa.reshape(-1, 2)
    cc = np.sin((bb[:, 0] + bb[:, 1]) * np.pi).reshape(-1, 1)

    X = np.concatenate([bb, cc], axis=1)

    if isinstance(model, BaseLDL):
        X_train, y_train = make_ldl()
        model.fit(X_train, y_train)
        y = model.predict(X)
    else:
        y = artificial(X, **kwargs)
        if isinstance(model, BaseLE):
            l = binaryzation(y)
            y = model.fit_transform(X, l)

    c = MinMaxScaler(feature_range=(1e-7, 1-1e-7)).fit_transform(y)
    colors = c.reshape(n_samples, n_samples, 3)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.xaxis._axinfo['grid']['linestyle'] = '--'
    ax.yaxis._axinfo['grid']['linestyle'] = '--'
    ax.zaxis._axinfo['grid']['linestyle'] = '--'
    ax.set_box_aspect(aspect=(1, 1, .2))
    ax.set_zticks([-1., 0., 1.])
    ax.axes.set_xlim3d((-1-1e-7, 1+1e-7))
    ax.axes.set_ylim3d((-1-1e-7, 1+1e-7))
    ax.axes.set_zlim3d((-1-1e-7, 1+1e-7))
    ax.xaxis.set_pane_color((1., 1., 1., 1.))
    ax.yaxis.set_pane_color((1., 1., 1., 1.))
    ax.zaxis.set_pane_color((1., 1., 1., 1.))
    ax.plot_surface(a1, a2, a3, facecolors=colors)
    
    if file_name is not None:
        if isinstance(file_name, str):
            fig.savefig(f'{file_name}.pdf', bbox_inches='tight')
        else:
            raise ValueError("Invalid file name, which should be a string.")
    else:
        plt.show()
