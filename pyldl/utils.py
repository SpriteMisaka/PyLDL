import os
import sys
import logging
import requests

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from pyldl.metrics import THE_SMALLER_THE_BETTER
from pyldl.algorithms.base import BaseLDL, BaseLE
from pyldl.algorithms.utils import normalize, proj, softmax, binaryzation


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


def load_dataset(name, dir='dataset'):
    if not os.path.exists(dir):
        logging.info(f'Directory {dir} does not exist, creating it.')
        os.makedirs(dir)
    dataset_path = os.path.join(dir, f'{name}.mat')
    if not os.path.exists(dataset_path):
        logging.info(f'Dataset {name}.mat does not exist, downloading it now, please wait...')
        download_dataset(name, dataset_path)
    data = sio.loadmat(dataset_path)
    return data['features'], data['labels']


def download_dataset(name, dataset_path):
    url = f'https://raw.githubusercontent.com/SpriteMisaka/PyLDL/main/dataset/{name}.mat'
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f'Failed to download {name}.mat')
    with open(dataset_path, 'wb') as f:
        f.write(response.content)
    logging.info(f'Dataset {name}.mat downloaded successfully.')


def gaussian_noise(D: np.ndarray, mean: float = 0., std: float = .1):
    return proj(D + np.random.normal(loc=mean, scale=std, size=D.shape))


def _random_mask(D, rate, weighted):
    if rate <= 0. or rate >= 1.:
        raise ValueError("Invalid rate, which should be in the range (0, 1).")
    if weighted:
        p = 1 - D
        p /= np.sum(p).flatten()
        select = np.random.choice(D.size, size=int(D.size * rate), replace=False, p=p)
        mask = np.zeros_like(D, dtype=bool)
        mask.flat[select] = True
    else:
        mask = np.random.rand(*D.shape) < rate
    return mask


def random_missing(D, rate=.8, weighted=False, return_mask=True):
    mask = _random_mask(D, rate, weighted)
    missing = D.copy()
    missing[mask] = .0
    return (missing, mask) if return_mask else missing


def random_exchange(D, rate=.2, weighted=False, return_mask=True):
    mask = _random_mask(D, rate, weighted)
    rows = np.where(np.sum(mask, axis=1) > 0)[0]
    exchanged = D.copy()
    for i in rows:
        idx = np.where(mask[i])[0]
        if len(idx) == 1:
            j = idx[0]
            k = np.random.choice([x for x in range(D.shape[1]) if x != j])
            exchanged[i, [j, k]] = exchanged[i, [k, j]]
        else:
            perm = idx.copy()
            while np.all(perm == idx):
                np.random.shuffle(perm)
            exchanged[i, idx] = exchanged[i, perm]
    return (exchanged, mask) if return_mask else exchanged


sys.modules['pyldl.utils.normalize'] = normalize
sys.modules['pyldl.utils.proj'] = proj
sys.modules['pyldl.utils.softmax'] = softmax
sys.modules['pyldl.utils.binaryzation'] = binaryzation


def emphasize(D, rate=.5, **kwargs):
    from scipy.special import softmax
    emphasized_D = D.copy()
    L = binaryzation(D, **kwargs)
    indices = np.random.choice(D.shape[0], size=int(D.shape[0] * rate), replace=False)
    for i in indices:
        n_pos = int(np.ceil(L[i].sum() / 2))
        where = np.where(L[i] == 1)[0]
        L[i] = 0
        select = np.random.choice(where.size, size=n_pos, replace=False)
        L[i, where[select]] = 1
        emphasized_D[i] += L[i]
        emphasized_D[i] = softmax(emphasized_D[i])
    return emphasized_D


def artificial(X, a=1., b=.5, c=.2, d=1.,
               w1=np.array([[4., 2., 1.]]),
               w2=np.array([[1., 2., 4.]]),
               w3=np.array([[1., 4., 2.]]),
               lambda1=.01, lambda2=.01):
    t = a * X + b * X**2 + c * X**3 + d
    psi1 = np.matmul(t, w1.T)**2
    psi2 = (np.matmul(t, w2.T) + lambda1 * psi1)**2
    psi3 = (np.matmul(t, w3.T) + lambda2 * psi2)**2
    D = np.concatenate([psi1, psi2, psi3], axis=1)
    return D / np.sum(D, axis=1).reshape(-1, 1)


def make_ldl(n_samples=200, random_state=None, **kwargs):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.uniform(-1, 1, (n_samples, 3))
    D = artificial(X, **kwargs)
    return X, D


def plot_artificial(n_samples=50, model=None, file_name=None, *,
                    noise=False, noise_func_args=None, **kwargs):

    if noise:
        noise_func_args = noise_func_args or [
            [lambda x: x, {}],
            [gaussian_noise, {}],
            [random_missing, {'rate': .5, 'return_mask': False}],
            [emphasize, {}]
        ]
    else:
        noise_func_args = [[lambda x: x, {}]]

    x1 = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    x2 = np.linspace(-1, 1, n_samples).reshape(-1, 1)

    a1, a2 = np.meshgrid(x1, x2)
    a3 = np.sin((a1 + a2) * np.pi)
    aa = np.concatenate([np.expand_dims(a1, axis=2),
                         np.expand_dims(a2, axis=2)], axis=2)
    bb = aa.reshape(-1, 2)
    cc = np.sin((bb[:, 0] + bb[:, 1]) * np.pi).reshape(-1, 1)

    X = np.concatenate([bb, cc], axis=1)

    colors = np.zeros((n_samples, n_samples, 3))

    n_batch = n_samples // len(noise_func_args)
    for i, n in enumerate(noise_func_args):
        start = i * n_batch
        end = (i + 1) * n_batch if i != len(noise_func_args) - 1 else n_samples

        if isinstance(model, BaseLDL):
            X_train, D_train = make_ldl()
            D_train = n[0](D_train, **n[1])
            model.fit(X_train, D_train)
            D = model.predict(X)
        else:
            D = artificial(X, **kwargs)
            if model is None:
                D = n[0](D, **n[1])
            elif isinstance(model, BaseLE):
                L = binaryzation(D)
                D = model.fit_transform(X, L)

        c = MinMaxScaler(feature_range=(1e-7, 1-1e-7)).fit_transform(D)
        colors[start:end,:,:] = c.reshape(n_samples, n_samples, 3)[start:end,:,:]

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.xaxis._axinfo['grid']['linestyle'] = '--'
    ax.yaxis._axinfo['grid']['linestyle'] = '--'
    ax.zaxis._axinfo['grid']['linestyle'] = '--'
    ax.set_box_aspect(aspect=(1, 1, .2))
    ax.set_xticks([-1., -.5, 0., .5, 1.])
    ax.set_yticks([-1., -.5, 0., .5, 1.])
    ax.set_zticks([-1., 0., 1.])
    ax.axes.set_xlim3d((-1-1e-7, 1+1e-7))
    ax.axes.set_ylim3d((-1-1e-7, 1+1e-7))
    ax.axes.set_zlim3d((-1-1e-7, 1+1e-7))
    ax.xaxis.set_pane_color((1., 1., 1., 1.))
    ax.yaxis.set_pane_color((1., 1., 1., 1.))
    ax.zaxis.set_pane_color((1., 1., 1., 1.))
    ax.plot_surface(a1, a2, a3, facecolors=colors)

    if file_name is None:
        plt.show()
    elif isinstance(file_name, str):
        fig.savefig(f'{file_name}.pdf', bbox_inches='tight')
    else:
        raise ValueError("Invalid file name, which should be a string.")


def regressor2ldl(regressor) -> BaseLDL:
    from pyldl.algorithms._problem_transformation import _Reg2LDL
    class Reg2LDL(_Reg2LDL):
        def _get_default_model(self):
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(regressor)
    return Reg2LDL()
