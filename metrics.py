import numpy as np
from scipy import stats
from sklearn.metrics import *
from sklearn.metrics.pairwise import *


eps = np.finfo(np.float64).eps


def chebyshev(y, y_pred):
    diff_abs = np.abs(y - y_pred)
    cheby = np.max(diff_abs, 1)

    return cheby.mean()


def clark(y, y_pred):
    y = np.clip(y, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    sum_2 = np.power(y + y_pred, 2)
    diff_2 = np.power(y - y_pred, 2)
    clark = np.sqrt(np.sum(diff_2 / sum_2, 1))
    
    return clark.mean()


def canberra(y, y_pred):
    y = np.clip(y, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    sum_2 = y + y_pred
    diff_abs = np.abs(y - y_pred)
    can = np.sum(diff_abs / sum_2, 1)
    
    return can.mean()


def kl_divergence(y, y_pred):
    y = np.clip(y, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)   
    kl = np.sum(y * (np.log(y) - np.log(y_pred)), 1)

    return kl.mean()


def cosine(y, y_pred):

    return 1 - paired_cosine_distances(y, y_pred).mean()


def intersection(y, y_pred):

    return 1 - 0.5 * np.sum(np.abs(y - y_pred), 1).mean()


def euclidean(y, y_pred):
    height = y.shape[0]

    return np.sum(np.sqrt(np.sum((y - y_pred) ** 2, 1))) / height


def sorensen(y, y_pred):
    height = y.shape[0]
    numerator = np.sum(np.abs(y - y_pred), 1)
    denominator = np.sum(y + y_pred, 1)

    return np.sum(numerator / denominator) / height


def squared_chi2(y, y_pred):
    height = y.shape[0]
    numerator = (y - y_pred) ** 2
    denominator = y + y_pred

    return np.sum(numerator / denominator) / height


def fidelity(y, y_pred):
    height = y.shape[0]

    return np.sum(np.sqrt(y * y_pred)) / height


def mean_absolute_error(y, y_pred, x=None):
    if x is None:
        x = np.arange(1, y.shape[1] + 1)
    yt = np.sum(y * x, axis=1)
    yp = np.sum(y_pred * x, axis=1)
    return np.average(np.abs(yt - yp))


def sort_loss(y, y_pred, reduction=np.average):
    i = np.argsort(-y)
    h = y_pred[np.arange(y_pred.shape[0])[:, np.newaxis], i]
    res = 0.
    for j in range(y.shape[1] - 1):
        for k in range(j + 1, y.shape[1]):
            res += np.maximum(h[:, k] - h[:, j], 0.) / np.log2(j + 2)
    res /= np.sum([1. / np.log2(j + 2) for j in range(y.shape[1] - 1)])
    return reduction(res) if reduction is not None else res


def spearman(y, y_pred):
    sum = 0.
    for i in range(y.shape[0]):
        s, _ = stats.spearmanr(y[i], y_pred[i])
        sum += s
    sum /= y.shape[0]
    return sum


def kendall(y, y_pred):
    sum = 0.
    for i in range(y.shape[0]):
        s, _ = stats.kendalltau(y[i], y_pred[i])
        sum += s
    sum /= y.shape[0]
    return sum


def score(y, y_pred,
          metrics=["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]):

    return tuple((eval(i)(y, y_pred) for i in metrics))
    