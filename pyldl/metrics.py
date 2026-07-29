import sys
from typing import Optional
from itertools import chain

import numpy as np
from scipy import stats

from pyldl.algorithms.utils import _clip, _reduction, _1d, kl_divergence, DEFAULT_METRICS, DEFAULT_METRICS_GLD
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support, roc_auc_score


sys.modules['pyldl.metrics.DEFAULT_METRICS'] = DEFAULT_METRICS

sys.modules['pyldl.metrics.DEFAULT_METRICS_GLD'] = DEFAULT_METRICS_GLD

_METRIC_REGISTRY = {}

def _register(prefix, larger=False, name=None):
    def decorator(func):
        _METRIC_REGISTRY[name or func.__name__] = (prefix, larger)
        return func
    return decorator

EPS = np.finfo(float).eps

LOG2 = np.log(2)


@_register("D")
@_reduction
@_1d
def chebyshev(D, D_pred):
    r"""Chebyshev distance. It is defined as:

    .. math::

        \text{Cheby.}(\boldsymbol{u}, \, \boldsymbol{v}) = \max_j \left\vert u_j - v_j \right\vert\text{.}
    """
    return np.max(np.abs(D - D_pred), 1)


@_register("D")
@_reduction
@_clip
@_1d
def clark(D, D_pred):
    r"""Clark distance. It is defined as:

    .. math::

        \text{Clark}(\boldsymbol{u}, \, \boldsymbol{v}) = \sqrt{\sum^l_{j=1}\frac{\left( u_j - v_j \right)^2}{\left( u_j + v_j \right)^2}}\text{.}
    """
    return np.sqrt(np.sum(np.power(D - D_pred, 2) / np.power(D + D_pred, 2), 1))


@_register("D")
@_reduction
@_clip
@_1d
def canberra(D, D_pred):
    r"""Canberra distance. It is defined as:

    .. math::

        \text{Can.}(\boldsymbol{u}, \, \boldsymbol{v}) = \sum^l_{j=1}\frac{\left\vert u_j - v_j \right\vert}{u_j + v_j}\text{.}
    """
    return np.sum(np.abs(D - D_pred) / (D + D_pred), 1)


sys.modules['pyldl.metrics.kl_divergence'] = kl_divergence
_register("D")(kl_divergence)


@_register("D")
@_reduction
@_clip
@_1d
def js_divergence(D, D_pred):
    r"""Jensen-Shannon divergence. It is defined as:

    .. math::

        \text{JSD}(\boldsymbol{u}, \, \boldsymbol{v}) = \frac{1}{2}\text{KLD}\left(\boldsymbol{u} \bigg\Vert \frac{1}{2}(\boldsymbol{u} + \boldsymbol{v}) \right) + \frac{1}{2}\text{KLD}\left(\boldsymbol{v} \bigg\Vert \frac{1}{2}(\boldsymbol{u} + \boldsymbol{v}) \right)\text{.}

    """
    M = .5 * (D + D_pred)
    return .5 * kl_divergence(D, M, reduction=None) + .5 * kl_divergence(D_pred, M, reduction=None)


@_register("D", larger=True)
@_reduction
@_1d
def cosine(D, D_pred):
    r"""Cosine similarity. It is defined as:

    .. math::

        \text{Cosine}(\boldsymbol{u}, \, \boldsymbol{v}) = \frac{\sum^l_{j=1}u_j v_j}{\sqrt{\sum^l_{j=1}u_j^2}\sqrt{\sum^l_{j=1}v_j^2}}\text{.}
    """
    from sklearn.metrics.pairwise import paired_cosine_distances
    return 1 - paired_cosine_distances(D, D_pred)


@_register("D", larger=True)
@_reduction
@_1d
def intersection(D, D_pred):
    r"""Intersection similarity. It is defined as:

    .. math::

        \text{Int.}(\boldsymbol{u}, \, \boldsymbol{v}) = \sum^l_{j=1} \min\left(u_j, \, v_j\right)\text{.}
    """
    return 1 - 0.5 * np.sum(np.abs(D - D_pred), 1)


@_register("D")
@_reduction
@_1d
def euclidean(D, D_pred):
    r"""Euclidean distance. It is defined as:

    .. math::

        \text{Eucl.}(\boldsymbol{u}, \, \boldsymbol{v}) = \sqrt{\sum^l_{j=1}\left( u_j - v_j \right)^2}\text{.}
    """
    return np.sqrt(np.sum((D - D_pred) ** 2, 1))


@_register("D")
@_reduction
@_clip
@_1d
def sorensen(D, D_pred):
    r"""
    .. raw:: html

        S&oslash;rensen's distance. It is defined as:

    .. math::

        \text{S}\phi\text{ren.}(\boldsymbol{u}, \, \boldsymbol{v}) = \frac{\sum^l_{j=1}\left\vert u_j - v_j \right\vert}{\sum^l_{j=1}\left( u_j + v_j \right)}\text{.}
    """
    return (np.sum(np.abs(D - D_pred), 1) / np.sum(D + D_pred, 1))


@_register("D")
@_reduction
@_clip
@_1d
def chi2(D, D_pred):
    r"""Chi-squared distance. It is defined as:

    .. math::

        \chi^2(\boldsymbol{u}, \, \boldsymbol{v}) = \sum^l_{j=1}\frac{\left( u_j - v_j \right)^2}{u_j + v_j}\text{.}
    """
    return np.sum((D - D_pred) ** 2 / (D + D_pred), 1)


@_register("D")
@_reduction
@_clip
@_1d
def wave_hedges(D, D_pred):
    r"""Wave-Hedges distance. It is defined as:

    .. math::

        \text{WHD}(\boldsymbol{u}, \, \boldsymbol{v}) = \sum^l_{j=1}\frac{\left| u_j - v_j \right|}{\max (u_j, \, v_j)}\text{.}
    """
    return np.sum(np.abs(D - D_pred) / np.maximum(D, D_pred), 1)


@_register("D", larger=True)
@_reduction
@_1d
def fidelity(D, D_pred):
    r"""Fidelity similarity. It is defined as:

    .. math::

        \text{Fid.}(\boldsymbol{u}, \, \boldsymbol{v}) = \sum^l_{j=1} \sqrt{u_j v_j}\text{.}
    """
    return np.sum(np.sqrt(D * D_pred), 1)


@_register("D", larger=True)
@_reduction
@_1d
def spearman(D, D_pred, transpose=False):
    r"""Spearman's rank correlation coefficient. It is defined as:

    .. math::

        \text{Spear.}(\boldsymbol{u}, \, \boldsymbol{v}) = 1 - \frac{6 \sum_{j=1}^{l} (\rho(u_j) - \rho(v_j))^2 }{l(l^2 - 1)}\text{,}

    where :math:`\rho(\cdot)` is the rank of the element in the vector.
    """
    D, D_pred = map(lambda X: np.transpose(X) if transpose else X, [D, D_pred])
    return np.array([stats.spearmanr(D[i], D_pred[i])[0] for i in range(D.shape[0])])


spearmanT = lambda G, G_pred: spearman(G, G_pred, transpose=True)
_register("G", larger=True, name="spearmanT")(spearmanT)


@_register("D", larger=True)
@_reduction
@_1d
def kendall(D, D_pred, transpose=False):
    r"""Kendall's rank correlation coefficient. It is defined as:

    .. math::

        \text{Ken.}(\boldsymbol{u}, \, \boldsymbol{v}) = \frac{2 \sum_{j < k} \text{sgn}(u_j - u_k) \text{sgn}(v_j - v_k) }{l (l-1)}\text{.}
    """
    D, D_pred = map(lambda X: np.transpose(X) if transpose else X, [D, D_pred])
    return np.array([stats.kendalltau(D[i], D_pred[i], variant='b')[0] for i in range(D.shape[0])])


kendallT = lambda G, G_pred: kendall(G, G_pred, transpose=True)
_register("G", larger=True, name="kendallT")(kendallT)


@_register("D", larger=True)
@_reduction
@_1d
def dpa(D, D_pred):
    r"""Degree percentile average (DPA) is proposed in paper :cite:`2024:jia`. It is defined as:

    .. math::

        \text{DPA}(\boldsymbol{u}, \, \boldsymbol{v}) = \frac{1}{l} \sum_{j=1}^{l} u_j \rho(v_j)\text{,}

    where :math:`\rho(\cdot)` is the rank of the element in the vector.
    """
    return np.mean(stats.rankdata(D_pred, axis=1) * D, axis=1)


def _uniform_vector(shape, scale=0.):
    u = 1 / shape[1] * np.ones(shape)
    if scale > 0:
        u += np.random.normal(0, scale, size=shape)
        u /= np.sum(u, axis=1, keepdims=True)
    return u


@_register("D", larger=True)
@_1d
def mu(D, D_pred):
    r"""The :math:`\mu` metric is proposed in paper :cite:`2025:li`. Its KL-divergence-based form is defined as:

    .. math::

        \mu(\boldsymbol{U}, \, \boldsymbol{V}) = \frac{1}{\delta_0} \int_0^{\delta_0} \frac{1}{n} \sum_{i=1}^{n} \mathbb{I} (\text{KLD}(\boldsymbol{u}_i, \, \boldsymbol{v}_i) \le \delta) \mathrm{d}\delta\text{,}

    where :math:`\delta_0 = \mathbb{E}_n[\text{KLD}(\boldsymbol{u}_i, \, \boldsymbol{c})]` and :math:`\boldsymbol{c}` is a uniform vector.
    """
    x0 = kl_divergence(D, _uniform_vector(D.shape))
    kld = kl_divergence(D, D_pred, reduction=None)
    return 1. - np.mean(np.minimum(kld, x0)) / x0


@_register("D")
@_reduction
@_1d
def divisiveness_error(D, D_pred, pos, neg):
    P, N = map(lambda x: np.reshape(x, (-1, 1)), [pos, neg])
    psi = lambda X: np.minimum(X @ P, X @ N)
    return np.abs(psi(D) - psi(D_pred))


def _mean(D, D_pred, op, mode='macro'):
    if mode == 'macro':
        if len(D.shape) == 2:
            D = np.sum(D * np.arange(1, D.shape[1] + 1), axis=1)
        if len(D_pred.shape) == 2:
            D_pred = np.sum(D_pred * np.arange(1, D_pred.shape[1] + 1), axis=1)
        return op(D, D_pred)
    elif mode == 'micro':
        return np.sum(op(D, D_pred), axis=1)


@_register("D")
@_reduction
def mean_absolute_error(D, D_pred, mode='macro'):
    return _mean(D, D_pred, lambda X, Y: np.abs(X - Y), mode=mode)


@_register("D")
@_reduction
def mean_squared_error(D, D_pred, mode='macro'):
    return _mean(D, D_pred, lambda X, Y: (X - Y) ** 2, mode=mode)


@_register("D")
@_reduction
@_1d
def zero_one_loss(D, D_pred):
    r"""0/1 loss. It is defined as:

    .. math::

        \text{0/1 loss}(\boldsymbol{u}, \, \boldsymbol{v}) = \delta(\arg\max(\boldsymbol{u}), \, \arg\max(\boldsymbol{v}))\text{,}

    where :math:`\delta(\cdot, \, \cdot)` is the Kronecker delta function.
    """
    return 1 - (np.argmax(D, 1) == np.argmax(D_pred, 1))


@_register("D")
@_reduction
@_1d
def error_probability(D, D_pred):
    r"""Error probability. It is defined as:

    .. math::

        \text{Err. prob.}(\boldsymbol{u}, \, \boldsymbol{v}) = 1 - u_{\arg\max(\boldsymbol{v})}\text{.}
    """
    return 1 - D[np.arange(D.shape[0]), np.argmax(D_pred, 1)]


@_register("G")
@_reduction
@_1d
def ood_error(G, G_pred):
    mask_ood = np.all(G <= 0., axis=1)
    pred_ood = np.all(G_pred <= 0., axis=1)
    results = np.zeros(G.shape[0], dtype=float)
    results[mask_ood] = ~pred_ood[mask_ood]
    results[~mask_ood] = np.argmax(G_pred[~mask_ood], axis=1) != np.argmax(G[~mask_ood], axis=1)
    return results


@_register("L")
@_reduction
@_1d
def hamming(L, L_pred):
    return np.mean(L != L_pred, 1)


@_register("L", larger=True)
@_reduction
@_1d
def jaccard(L, L_pred):
    return np.sum(np.logical_and(L, L_pred), 1) / (np.sum(np.logical_or(L, L_pred), 1) + EPS)


@_register("L", larger=True)
@_reduction
@_1d
def subset_accuracy(L, L_pred):
    return np.all(L == L_pred, 1).astype(float)


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


@_register("D", larger=True)
def match_m(D, D_pred, m=None):
    if m is None:
        m = [1, 2, 3, 4]
    return _calculate_match_m_top_k(D, D_pred, m, 'match_m')


@_register("D", larger=True)
def top_k(D, D_pred, k=None, mode='f1_score'):
    if k is None:
        k = [1, 2, 3, 4]
    return _calculate_match_m_top_k(D, D_pred, k, 'top_k', mode)


@_register("D", larger=True)
def max_roc_auc(D, D_pred):
    L = np.eye(D.shape[1])[np.argmax(D, 1)]
    L_pred = np.eye(D_pred.shape[1])[np.argmax(D_pred, 1)]
    return roc_auc_score(L.ravel(), L_pred.ravel())


@_register("Y", larger=True)
def precision(y, y_pred):
    return precision_score(y, y_pred, average='macro')


@_register("Y", larger=True)
def specificity(y, y_pred):
    result = 0.
    labels = np.union1d(np.unique(y), np.unique(y_pred))
    for i in labels:
        _, recall, _, _ = precision_recall_fscore_support(
            y==i, y_pred==i, pos_label=True, average=None
        )
        result += recall[0]
    return result / len(labels)


@_register("Y", larger=True)
def sensitivity(y, y_pred):
    return recall_score(y, y_pred, average='macro')


@_register("Y", larger=True)
def youden_index(y, y_pred):
    return sensitivity(y, y_pred) + specificity(y, y_pred) - 1


@_register("Y", larger=True)
def accuracy(y, y_pred):
    return accuracy_score(y, y_pred)


def worst_kl_divergence(D: np.ndarray):
    from pyldl.algorithms.utils import estimate_alpha
    from scipy.special import digamma
    alpha = estimate_alpha(D)
    alpha_sum = np.sum(alpha)
    return np.sum(
        alpha * ((digamma(alpha + 1) - digamma(alpha_sum + 1)) + np.log(len(alpha)))
    ) / alpha_sum


_TYPES = "DLGY"
_GROUPED = {p: [[], []] for p in _TYPES}
for name, (prefix, larger) in _METRIC_REGISTRY.items():
    _GROUPED[prefix][1 if larger else 0].append(name)

for prefix, (smaller, larger) in _GROUPED.items():
    globals()[f"{prefix}_METRICS_THE_SMALLER_THE_BETTER"] = tuple(smaller)
    globals()[f"{prefix}_METRICS_THE_LARGER_THE_BETTER"] = tuple(larger)
    globals()[f"{prefix}_METRICS"] = tuple(smaller + larger)

THE_SMALLER_THE_BETTER = tuple(chain.from_iterable(_GROUPED[p][0] for p in _TYPES))
THE_LARGER_THE_BETTER = tuple(chain.from_iterable(_GROUPED[p][1] for p in _TYPES))

NAME_TO_TYPE = {name: prefix for name, (prefix, _) in _METRIC_REGISTRY.items()}


def score(target: np.ndarray, pred: np.ndarray,
          metrics: Optional[list] = None, return_dict: bool = False):
    if metrics is None:
        metrics = DEFAULT_METRICS
    scores = tuple((eval(i)(target, pred) if isinstance(i, str) else i(target, pred) for i in metrics))
    return dict(zip(metrics, scores)) if return_dict else scores
