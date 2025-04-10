from typing import Optional
import sys

import numpy as np
from scipy import stats

from pyldl.algorithms.utils import _clip, _reduction, kl_divergence, sort_loss, DEFAULT_METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support, roc_auc_score


THE_SMALLER_THE_BETTER = ["chebyshev", "clark", "canberra", "kl_divergence",
                          "euclidean", "sorensen", "squared_chi2",
                          "mean_absolute_error", "mean_squared_error",
                          "sort_loss",
                          "zero_one_loss", "error_probability"]

THE_LARGER_THE_BETTER = ["cosine", "intersection",
                         "fidelity",
                         "spearman", "kendall", "dpa",
                         "match_m", "top_k", "max_roc_auc",
                         "precision", "specificity", "sensitivity", "youden_index", "accuracy"]

sys.modules['pyldl.metrics.DEFAULT_METRICS'] = DEFAULT_METRICS


@_reduction
def chebyshev(D, D_pred):
    """Chebyshev distance. It is defined as:

    .. math::

        \\text{Cheby.}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = \\max_j \\left\\vert u_j - v_j \\right\\vert\\text{.}
    """
    return np.max(np.abs(D - D_pred), 1)


@_reduction
@_clip
def clark(D, D_pred):
    """Clark distance. It is defined as:

    .. math::

        \\text{Clark}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = \\sqrt{\\sum^l_{j=1}\\frac{\\left( u_j - v_j \\right)^2}{\\left( u_j + v_j \\right)^2}}\\text{.}
    """
    return np.sqrt(np.sum(np.power(D - D_pred, 2) / np.power(D + D_pred, 2), 1))


@_reduction
@_clip
def canberra(D, D_pred):
    """Canberra distance. It is defined as:

    .. math::

        \\text{Can.}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = \\sum^l_{j=1}\\frac{\\left\\vert u_j - v_j \\right\\vert}{u_j + v_j}\\text{.}
    """
    return np.sum(np.abs(D - D_pred) / (D + D_pred), 1)


sys.modules['pyldl.metrics.kl_divergence'] = kl_divergence


@_reduction
def cosine(D, D_pred):
    """Cosine similarity. It is defined as:

    .. math::

        \\text{Cosine}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = \\frac{\\sum^l_{j=1}u_j v_j}{\\sqrt{\\sum^l_{j=1}u_j^2}\\sqrt{\\sum^l_{j=1}v_j^2}}\\text{.}
    """
    from sklearn.metrics.pairwise import paired_cosine_distances
    return 1 - paired_cosine_distances(D, D_pred)


@_reduction
def intersection(D, D_pred):
    """Intersection similarity. It is defined as:

    .. math::

        \\text{Int.}(\\boldsymbol{u}, \, \\boldsymbol{v}) = \\sum^l_{j=1} \\min\\left(u_j, \\, v_j\\right)\\text{.}
    """
    return 1 - 0.5 * np.sum(np.abs(D - D_pred), 1)


@_reduction
def euclidean(D, D_pred):
    return np.sqrt(np.sum((D - D_pred) ** 2, 1))


@_reduction
@_clip
def sorensen(D, D_pred):
    return (np.sum(np.abs(D - D_pred), 1) / np.sum(D + D_pred, 1))


@_reduction
@_clip
def squared_chi2(D, D_pred):
    return np.sum((D - D_pred) ** 2 / (D + D_pred), 1)


@_reduction
def fidelity(D, D_pred):
    return np.sum(np.sqrt(D * D_pred), 1)


sys.modules['pyldl.metrics.sort_loss'] = sort_loss


@_reduction
def spearman(D, D_pred):
    """Spearman's rank correlation coefficient. It is defined as:

    .. math::

        \\text{Spear.}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = 1 - \\frac{6 \\sum_{j=1}^{l} (\\rho(u_j) - \\rho(v_j))^2 }{l(l^2 - 1)}\\text{,}

    where :math:`\\rho` is the rank of the element in the vector.
    """
    return np.array([stats.spearmanr(D[i], D_pred[i])[0] for i in range(D.shape[0])])


@_reduction
def kendall(D, D_pred):
    """Kendall's rank correlation coefficient. It is defined as:

    .. math::

        \\text{Ken.}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = \\frac{2 \\sum_{j < k} \\text{sgn}(u_j - u_k) \\text{sgn}(v_j - v_k) }{l (l-1)}\\text{.}
    """
    return np.array([stats.kendalltau(D[i], D_pred[i], variant='c')[0] for i in range(D.shape[0])])


@_reduction
def dpa(D, D_pred):
    return np.mean(stats.rankdata(D_pred, axis=1) * D, axis=1)


@_reduction
def mean_absolute_error(D, D_pred, mode='macro'):
    if mode == 'macro':
        if len(D.shape) == 2:
            D = np.sum(D * np.arange(1, D.shape[1] + 1), axis=1)
        if len(D_pred.shape) == 2:
            D_pred = np.sum(D_pred * np.arange(1, D_pred.shape[1] + 1), axis=1)
        return np.abs(D - D_pred)
    elif mode == 'micro':
        return np.sum(np.abs(D - D_pred), axis=1)


@_reduction
def mean_squared_error(D, D_pred, mode='macro'):
    if mode == 'macro':
        if len(D.shape) == 2:
            D = np.sum(D * np.arange(1, D.shape[1] + 1), axis=1)
        if len(D_pred.shape) == 2:
            D_pred = np.sum(D_pred * np.arange(1, D_pred.shape[1] + 1), axis=1)
        return (D - D_pred) ** 2
    elif mode == 'micro':
        return np.sum((D - D_pred) ** 2, axis=1)


@_reduction
def zero_one_loss(D, D_pred):
    """0/1 loss. It is defined as:

    .. math::

        \\text{0/1 loss}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = \\delta(\\arg\\max(\\boldsymbol{u}), \\, \\arg\\max(\\boldsymbol{v}))\\text{,}

    where :math:`\\delta` is the Kronecker delta function.
    """
    return 1 - (np.argmax(D, 1) == np.argmax(D_pred, 1))


@_reduction
def error_probability(D, D_pred):
    """Error probability. It is defined as:

    .. math::

        \\text{Err. prob.}(\\boldsymbol{u}, \\, \\boldsymbol{v}) = 1 - u_{\\arg\\max(\\boldsymbol{v})}\\text{.}
    """
    return 1 - D[np.arange(D.shape[0]), np.argmax(D_pred, 1)]


def _calculate_match_m_top_k(D, D_pred, params, mode, top_k_mode='f1_score'):
    if type(params) == int:
        params = [params]
    results = []
    for param in params:
        scores = []
        for d, d_pred in zip(D, D_pred):
            top = set(np.argsort(d)[-param:])
            top_pred = set(np.argsort(d_pred)[-param:])
            intersection = top.intersection(top_pred)
            if mode == 'match_m':
                scores.append(len(intersection) / min(param, len(d)))
            elif mode == 'top_k':
                p = len(intersection) / param
                if top_k_mode == 'precision':
                    scores.append(p)
                    continue
                r = len(intersection) / len(d)
                if top_k_mode == 'recall':
                    scores.append(r)
                    continue
                f1 = 2 * p * r / (p + r) if p + r > 0 else 0
                if top_k_mode == 'f1_score':
                    scores.append(f1)
        results.append(np.average(scores))
    return results[0] if len(results) == 1 else results


def match_m(D, D_pred, m=None):
    if m is None:
        m = [1, 2, 3, 4]
    return _calculate_match_m_top_k(D, D_pred, m, 'match_m')


def top_k(D, D_pred, k=None, mode='f1_score'):
    if k is None:
        k = [1, 2, 3, 4]
    return _calculate_match_m_top_k(D, D_pred, k, 'top_k', mode)


def max_roc_auc(D, D_pred):
    L = np.eye(D.shape[1])[np.argmax(D, 1)]
    L_pred = np.eye(D_pred.shape[1])[np.argmax(D_pred, 1)]
    return roc_auc_score(L.ravel(), L_pred.ravel())


def precision(y, y_pred):
    return precision_score(y, y_pred, average='macro')


def specificity(y, y_pred):
    result = 0.
    labels = np.union1d(np.unique(y), np.unique(y_pred))
    for i in labels:
        _, recall, _, _ = precision_recall_fscore_support(
            y==i, y_pred==i, pos_label=True, average=None
        )
        result += recall[0]
    return result / len(labels)


def sensitivity(y, y_pred):
    return recall_score(y, y_pred, average='macro')


def youden_index(y, y_pred):
    return sensitivity(y, y_pred) + specificity(y, y_pred) - 1


def accuracy(y, y_pred):
    return accuracy_score(y, y_pred)


def score(target: np.ndarray, pred: np.ndarray,
          metrics: Optional[list] = None, return_dict: bool = False):
    if metrics is None:
        metrics = DEFAULT_METRICS
    scores = tuple((eval(i)(target, pred) if isinstance(i, str) else i(target, pred) for i in metrics))
    return dict(zip(metrics, scores)) if return_dict else scores
