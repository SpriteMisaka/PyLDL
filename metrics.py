import numpy as np
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


def score(y, y_pred,
          metrics=["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]):

    return tuple((eval(i)(y, y_pred) for i in metrics))
    