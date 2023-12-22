import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from algorithms import _BaseLDL, _BaseLE


def load_dataset(name):
    data = sio.loadmat('dataset/' + name)
    return data['features'], data['labels']


def random_missing(y, w=.1):
    missing_y = y.copy()
    np.put(missing_y,
           np.random.choice(y.size, int(y.size * w), replace=False),
           np.nan)
    missing_mask = np.isnan(missing_y)
    missing_y[missing_mask] = 0.
    return missing_y, missing_mask


def binaryzation(y, t=.5):
    b = np.sort(y.T, axis=0)[::-1]
    cs = np.cumsum(b, axis=0)
    m = np.argmax(cs > t, axis=0)
    r = np.argsort(np.argsort(y))
    return np.where(r >= y.shape[1] - m.reshape(-1, 1) - 1, 1, 0)


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


def make_ldl(n_samples=200):
    X = np.random.uniform(-1, 1, (n_samples, 3))
    y = artificial(X)
    return X, y


def plot_artificial(n_samples=50, model=None, figname='output'):

    x1 = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    x2 = np.linspace(-1, 1, n_samples).reshape(-1, 1)

    a1, a2 = np.meshgrid(x1, x2)
    a3 = np.sin((a1 + a2) * np.pi)
    aa = np.concatenate([np.expand_dims(a1, axis=2),
                         np.expand_dims(a2, axis=2)], axis=2)
    bb = aa.reshape(-1, 2)
    cc = np.sin((bb[:, 0] + bb[:, 1]) * np.pi).reshape(-1, 1)

    X = np.concatenate([bb, cc], axis=1)

    if isinstance(model, _BaseLDL):
        X_train, y_train = make_ldl()
        model.fit(X_train, y_train)
        y = model.predict(X)
    else:
        y = artificial(X)
        if isinstance(model, _BaseLE):
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
    
    fig.savefig(f'{figname}.pdf', bbox_inches='tight')
