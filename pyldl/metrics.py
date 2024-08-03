from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import paired_cosine_distances


eps = np.finfo(np.float64).eps


THE_SMALLER_THE_BETTER = ["chebyshev", "clark", "canberra", "kl_divergence",
                          "euclidean", "sorensen", "squared_chi2",
                          "mean_absolute_error",
                          "sort_loss",
                          "zero_one_loss", "error_probability"]

THE_LAGER_THE_BETTER = ["cosine", "intersection",
                        "fidelity",
                        "spearman", "kendall", "dpa"]

DEFAULT_METRICS = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]


def _clip(func):
    def _wrapper(y, y_pred):
        y = np.clip(y, eps, 1)
        y_pred = np.clip(y_pred, eps, 1)
        return func(y, y_pred)
    return _wrapper


def _reduction(func):
    def _wrapper(*args, reduction=np.average):
        results = func(*args)
        return reduction(results) if reduction is not None else results
    return _wrapper


@_reduction
def chebyshev(y, y_pred):
    return np.max(np.abs(y - y_pred), 1)


@_reduction
@_clip
def clark(y, y_pred):
    return np.sqrt(np.sum(np.power(y - y_pred, 2) / np.power(y + y_pred, 2), 1))


@_reduction
@_clip
def canberra(y, y_pred):
    return np.sum(np.abs(y - y_pred) / (y + y_pred), 1)


@_reduction
@_clip
def kl_divergence(y, y_pred):
    return np.sum(y * (np.log(y) - np.log(y_pred)), 1)


@_reduction
def cosine(y, y_pred):
    return 1 - paired_cosine_distances(y, y_pred)


@_reduction
def intersection(y, y_pred):
    return 1 - 0.5 * np.sum(np.abs(y - y_pred), 1)


@_reduction
def euclidean(y, y_pred):
    return np.sqrt(np.sum((y - y_pred) ** 2, 1))


@_reduction
@_clip
def sorensen(y, y_pred):
    return (np.sum(np.abs(y - y_pred), 1) / np.sum(y + y_pred, 1))


@_reduction
@_clip
def squared_chi2(y, y_pred):
    return np.sum((y - y_pred) ** 2 / (y + y_pred), 1)


@_reduction
def fidelity(y, y_pred):
    return np.sum(np.sqrt(y * y_pred), 1)


@_reduction
def mean_absolute_error(y, y_pred):
    x = np.arange(1, y.shape[1] + 1)
    yt = np.sum(y * x, axis=1)
    yp = np.sum(y_pred * x, axis=1)
    return np.abs(yt - yp)


@_reduction
def sort_loss(y, y_pred):
    i = np.argsort(-y)
    h = y_pred[np.arange(y_pred.shape[0])[:, np.newaxis], i]
    res = 0.
    for j in range(y.shape[1] - 1):
        for k in range(j + 1, y.shape[1]):
            res += np.maximum(h[:, k] - h[:, j], 0.) / np.log2(j + 2)
    res /= np.sum([1. / np.log2(j + 2) for j in range(y.shape[1] - 1)])
    return res


@_reduction
def spearman(y, y_pred):
    return np.array([stats.spearmanr(y[i], y_pred[i])[0] for i in range(y.shape[0])])


@_reduction
def kendall(y, y_pred):
    return np.array([stats.kendalltau(y[i], y_pred[i], variant='c')[0] for i in range(y.shape[0])])


@_reduction
def zero_one_loss(y, y_pred):
    return 1 - (np.argmax(y, 1) == np.argmax(y_pred, 1))


@_reduction
def error_probability(y, y_pred):
    return 1 - y[np.arange(y.shape[0]), np.argmax(y_pred, 1)]


@_reduction
def dpa(y, y_pred):
    return np.mean(stats.rankdata(y_pred, axis=1) * y, axis=1)


def score(y: np.ndarray, y_pred: np.ndarray,
          metrics: Optional[list[str]] = None, return_dict: bool = False):
    if metrics is None:
        metrics = DEFAULT_METRICS
    scores = tuple((eval(i)(y, y_pred) for i in metrics))
    return dict(zip(metrics, scores)) if return_dict else scores
