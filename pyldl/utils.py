import os
import sys
import logging
import requests

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pyldl.algorithms.base import BaseLDL, BaseLE, BaseGLD
from pyldl.algorithms.utils import normalize, proj, softmax, binaryzation


def load_dataset(name, dir='datasets', mode='D'):
    r"""Load a dataset from a local ``.mat`` file, downloading it when necessary. 
    If the dataset is not found in the directory, and it is available in the PyLDL GitHub repository, it will be downloaded automatically.

    :param name: Name of the dataset (without the ``.mat`` extension).
    :type name: str
    :param dir: Directory where the dataset is stored (or will be downloaded to), defaults to 'datasets'.
    :type dir: str
    :param mode: Mode of the dataset to load, defaults to 'D'. 
        'D' is for label distribution, and 'G' is for generalized label distribution.
    :type mode: {'D', 'G'}
    """
    if not os.path.exists(dir):
        logging.info(f'Directory {dir} does not exist, creating it.')
        os.makedirs(dir)
    dataset_path = os.path.join(dir, f'{name}.mat')
    if not os.path.exists(dataset_path):
        logging.info(f'Dataset {name}.mat does not exist, downloading it now, please wait...')
        download_dataset(name, dataset_path)

    def _data2G(data):
        Q = data['raw_data']
        lb = data.get('lower_bounds', Q.min(axis=0).reshape(1, -1))
        ub = data.get('upper_bounds', Q.max(axis=0).reshape(1, -1))
        G = BaseGLD.RAW2G(Q, lb, ub)
        return G, lb, ub

    data = sio.loadmat(dataset_path)
    X = data['features']
    if mode == 'D':
        D = data.get('labels')
        if D is None:
            D = BaseGLD.G2D(*_data2G(data))
        return X, D
    elif mode == 'G':
        return X, *_data2G(data)
    else:
        raise ValueError(f"Unknown mode: {mode}. "
                         "Supported modes are 'D' for label distribution and 'G' for generalized label distribution.")


def download_dataset(name, dataset_path):
    r"""Download a named ``.mat`` dataset from the PyLDL GitHub repository.

    :param name: Name of the dataset (without the ``.mat`` extension).
    :type name: str
    :param dataset_path: Local path where the downloaded dataset is saved.
    :type dataset_path: str
    """
    url = f'https://raw.githubusercontent.com/SpriteMisaka/PyLDL/main/datasets/{name}.mat'
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f'Failed to download {name}.mat')
    with open(dataset_path, 'wb') as f:
        f.write(response.content)
    logging.info(f'Dataset {name}.mat downloaded successfully.')


def gaussian_noise(D: np.ndarray, mode=None, mean: float = 0., std: float = .1, p: float = 5.):
    r"""Add Gaussian noise to label distributions and renormalize them.

    :param D: Label distribution matrix.
    :type D: np.ndarray
    :param mode: Noise mode, defaults to None. 
        Mode 'gcia' refers to *Generative Calibration of Inaccurate Annotators*. See the following reference for details:

        .. bibliography:: bib/others/noisy/references.bib
            :filter: False
            :labelprefix: NOISE-
            :keyprefix: noise-

            2024:he

    :type mode: {None, 'gcia'}
    :param mean: Mean of the noise in the default mode, defaults to 0.
    :type mean: float
    :param std: Standard deviation of the noise in the default mode, defaults to 0.1.
    :type std: float
    :param p: Annotator professionalism level in GCIA mode, defaults to 5.; larger values produce more noise.
    :type p: float
    :return: Noisy label distribution matrix.
    :rtype: np.ndarray
    """
    if mode is None:
        return proj(D + np.random.normal(loc=mean, scale=std, size=D.shape))
    elif mode == 'gcia':
        if p <= 0:
            raise ValueError("p must be positive.")
        variance = D / D.shape[1] * p
        return proj(np.random.normal(loc=D, scale=np.sqrt(variance)))
    else:
        raise ValueError(f"Unknown mode: {mode}.")


def _random_mask(D, rate, weighted):
    if rate <= 0. or rate >= 1.:
        raise ValueError("Invalid rate, which should be in the range (0, 1).")
    return _weighted_mask(D, rate) if weighted else np.random.rand(*D.shape) < rate


def _weighted_mask(D, rate):
    p = 1 - D
    p /= np.sum(p)
    p = p.flatten()
    select = np.random.choice(D.size, size=int(D.size * rate), replace=False, p=p)
    result = np.zeros_like(D, dtype=bool)
    result.flat[select] = True
    return result


def random_missing(D, rate=.8, weighted=False, return_mask=True):
    r"""Randomly set entries of label distributions to zero.
    This setting is useful for simulating incomplete label distributions. See :cite:`2017:xu` for details.

    :param D: Label distribution matrix.
    :type D: np.ndarray
    :param rate: Proportion of entries to remove, in the range (0, 1), defaults to 0.8.
    :type rate: float
    :param weighted: Whether to select entries according to value-based weights, defaults to False.
        This setting can refer to:

        .. bibliography:: bib/incomldl/references.bib
            :filter: False
            :labelprefix: NOISE-
            :keyprefix: noise-

            2024:li

    :type weighted: bool
    :param return_mask: Whether to return the missing-entry mask, defaults to True.
    :type return_mask: bool
    :return: The incomplete distribution, optionally together with its boolean mask.
    :rtype: np.ndarray or tuple[np.ndarray, np.ndarray]
    """
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
    r"""Generate artificial label distributions from polynomial feature interactions. 
    The generation process is provided in :cite:`2016:geng`.

    :param X: Feature matrix.
    :type X: np.ndarray
    :param a: Coefficient of the linear term, defaults to 1.
    :param b: Coefficient of the quadratic term, defaults to 0.5.
    :param c: Coefficient of the cubic term, defaults to 0.2.
    :param d: Constant term, defaults to 1.
    :param w1: Weights for the first label, defaults to ``[[4., 2., 1.]]``.
    :param w2: Weights for the second label, defaults to ``[[1., 2., 4.]]``.
    :param w3: Weights for the third label, defaults to ``[[1., 4., 2.]]``.
    :param lambda1: Coupling coefficient for the second label, defaults to 0.01.
    :param lambda2: Coupling coefficient for the third label, defaults to 0.01.
    :return: Artificial label distribution matrix.
    :rtype: np.ndarray
    """
    t = a * X + b * X**2 + c * X**3 + d
    psi1 = np.matmul(t, w1.T)**2
    psi2 = (np.matmul(t, w2.T) + lambda1 * psi1)**2
    psi3 = (np.matmul(t, w3.T) + lambda2 * psi2)**2
    D = np.concatenate([psi1, psi2, psi3], axis=1)
    return D / np.sum(D, axis=1).reshape(-1, 1)


def make_ldl(n_samples=200, random_state=None, **kwargs):
    r"""Generate random features and artificial label distributions using the :func:`artificial` function.

    :param n_samples: Number of samples to generate, defaults to 200.
    :type n_samples: int
    :param random_state: Seed for NumPy's random number generator, defaults to None.
    :type random_state: int or None
    :param kwargs: Additional keyword arguments passed to :func:`artificial`.
    :return: Generated feature matrix and label distribution matrix.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1, 1, (n_samples, 3))
    D = artificial(X, **kwargs)
    return X, D


def plot_artificial(grid_size=50, model=None, file_name=None, *,
                    noise=False, noise_func_args=None, **kwargs):
    r"""Plot predictions or perturbations on the artificial LDL dataset as a 3D color surface.

    :param grid_size: Number of points along each feature axis, defaults to 50.
    :type grid_size: int
    :param model: Optional LDL or LE model used to generate predictions, defaults to None.
    :type model: BaseLDL or BaseLE or None
    :param file_name: Output file name without extension; if None, display the figure, defaults to None.
    :type file_name: str or None
    :param noise: Whether to apply the default noise transformations, defaults to False.
    :type noise: bool
    :param noise_func_args: Noise functions and their keyword arguments, defaults to None.
    :type noise_func_args: list or None
    :param kwargs: Additional keyword arguments passed to :func:`artificial`.
    """

    if noise:
        noise_func_args = noise_func_args or [
            [lambda x: x, {}],
            [gaussian_noise, {}],
            [random_missing, {'rate': .5, 'return_mask': False}],
            [emphasize, {}]
        ]
    else:
        noise_func_args = [[lambda x: x, {}]]

    x1 = np.linspace(-1, 1, grid_size).reshape(-1, 1)
    x2 = np.linspace(-1, 1, grid_size).reshape(-1, 1)

    a1, a2 = np.meshgrid(x1, x2)
    a3 = np.sin((a1 + a2) * np.pi)
    aa = np.concatenate([np.expand_dims(a1, axis=2),
                         np.expand_dims(a2, axis=2)], axis=2)
    bb = aa.reshape(-1, 2)
    cc = np.sin((bb[:, 0] + bb[:, 1]) * np.pi).reshape(-1, 1)

    X = np.concatenate([bb, cc], axis=1)

    colors = np.zeros((grid_size, grid_size, 3))

    n_batch = grid_size // len(noise_func_args)
    for i, n in enumerate(noise_func_args):
        start = i * n_batch
        end = (i + 1) * n_batch if i != len(noise_func_args) - 1 else grid_size

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

        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        hsv = rgb_to_hsv(D.reshape(-1, 1, 3))
        hsv[:, :, 1] *= 1.75
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0., 1.)
        c = hsv_to_rgb(hsv) ** .4545
        colors[start:end, :, :] = c.reshape(grid_size, grid_size, 3)[start:end, :, :]

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
    r"""Wrap a scikit-learn regressor as a PyLDL label distribution learner.

    :param regressor: Scikit-learn regressor used for multi-output prediction.
    :type regressor: sklearn.base.RegressorMixin
    :return: A PyLDL model backed by the supplied regressor.
    :rtype: BaseLDL
    """
    from pyldl.algorithms._problem_transformation import _Reg2LDL
    class Reg2LDL(_Reg2LDL):
        def _get_default_model(self):
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(regressor)
    return Reg2LDL()
