from typing import Optional
from inspect import signature
from functools import wraps, singledispatch

import numpy as np


EPS = np.finfo(np.float64).eps

DEFAULT_METRICS = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]

DEFAULT_METRICS_GLD = ['clark', 'mu', 'hamming', 'subset_accuracy', 'spearman', 'kendallT', 'ood_error']


def _clip(func):

    @singledispatch
    def _wrapper(D, D_pred, **kwargs):
        import keras.ops as ops
        D = ops.clip(D, EPS, 1.)
        D_pred = ops.clip(D_pred, EPS, 1.)
        return func(D, D_pred, **kwargs)

    @_wrapper.register(np.ndarray)
    def _(D: np.ndarray, D_pred: np.ndarray, **kwargs):
        D = np.clip(D, EPS, 1.)
        D_pred = np.clip(D_pred, EPS, 1.)
        return func(D, D_pred, **kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _wrapper(*args, **kwargs)
    return wrapper


def _reduction(func):
    @wraps(func)
    def _wrapper(*args, reduction=np.average, **kwargs):
        results = func(*args, **kwargs)
        return reduction(results) if reduction is not None else results
    return _wrapper


def _1d(func):

    @singledispatch
    def _wrapper(X, Y=None, **kwargs):
        import keras.ops as ops
        if len(ops.shape(X)) != 1:
            return func(X, Y, **kwargs)
        X = ops.expand_dims(X, axis=0)
        if Y is not None:
            Y = ops.expand_dims(Y, axis=0)
        results = func(X, Y, **kwargs)
        return ops.squeeze(results, axis=0)

    @_wrapper.register(np.ndarray)
    def _(X: np.ndarray, Y: Optional[np.ndarray] = None, **kwargs):
        if len(np.shape(X)) != 1:
            return func(X, Y, **kwargs)
        X = np.expand_dims(X, axis=0)
        if Y is not None:
            Y = np.expand_dims(Y, axis=0)
        results = func(X, Y, **kwargs)
        return np.squeeze(results, axis=0)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _wrapper(*args, **kwargs)
    return wrapper


@_reduction
@_clip
@_1d
def kl_divergence(D, D_pred):
    r"""Kullback-Leibler divergence. It is defined as:

    .. math::

        \text{KLD}(\boldsymbol{u}, \, \boldsymbol{v}) = \sum^l_{j=1}u_j \ln \frac{u_j}{v_j}\text{.}
    """
    return np.sum(D * (np.log(D) - np.log(D_pred)), 1)


@_reduction
@_1d
def sort_loss(D, D_pred):
    i = np.argsort(-D)
    h = D_pred[np.arange(D_pred.shape[0])[:, np.newaxis], i]
    res = 0.
    for j in range(D.shape[1] - 1):
        for k in range(j + 1, D.shape[1]):
            res += np.maximum(h[:, k] - h[:, j], 0.) / np.log2(j + 2)
    res /= np.sum([1. / np.log2(j + 2) for j in range(D.shape[1] - 1)])
    return res


def soft_thresholding(A: np.ndarray, tau: float) -> np.ndarray:
    r"""Soft thresholding operation.
    It is defined as :math:`\text{soft}(\boldsymbol{A}, \, \tau) = \text{sgn}(\boldsymbol{A}) \odot \max\lbrace \lvert \boldsymbol{A} \rvert - \tau, 0 \rbrace`, 
    where :math:`\odot` denotes element-wise multiplication.

    :param A: Matrix :math:`\boldsymbol{A}`.
    :type A: np.ndarray
    :param tau: :math:`\tau`.
    :type tau: float
    :return: The result of soft thresholding operation.
    :rtype: np.ndarray
    """
    return np.sign(A) * np.maximum(np.abs(A) - tau, 0.)


def svt(A: np.ndarray, tau: float) -> np.ndarray:
    r"""Singular value thresholding (SVT) is proposed in paper :cite:`2010:cai`.

    The solution to the optimization problem 
    :math:`\mathop{\arg\min}_{\boldsymbol{X}} \Vert \boldsymbol{X} - \boldsymbol{A} \Vert_\text{F}^2 + \tau \Vert \boldsymbol{X} \Vert_{\ast}` 
    is given by :math:`\boldsymbol{U} \max \lbrace \boldsymbol{\Sigma} - \tau, 0 \rbrace \boldsymbol{V}^\top`, 
    where :math:`\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top` is the singular value decomposition of matrix :math:`\boldsymbol{A}`.

    :param A: Matrix :math:`\boldsymbol{A}`.
    :type A: np.ndarray
    :param tau: :math:`\tau`.
    :type tau: float
    :return: The solution to the optimization problem.
    :rtype: np.ndarray
    """
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0.)
    return U @ np.diag(S_thresh) @ VT


def solvel21(A: np.ndarray, tau: float) -> np.ndarray:
    r"""This approach is proposed in paper :cite:`2014:chen`.

    The solution to the optimization problem 
    :math:`\mathop{\arg\min}_{\boldsymbol{X}} \Vert \boldsymbol{X} - \boldsymbol{A} \Vert_\text{F}^2 + \tau \Vert \boldsymbol{X} \Vert_{2,\,1}` 
    is given by the following formula: 

    .. math::

        \vec{x}_{\bullet j}^{\ast} = \left\{
        \begin{aligned}
        & \frac{\Vert \vec{a}_{\bullet j} \Vert - \tau}{\Vert \vec{a}_{\bullet j} \Vert} \vec{a}_{\bullet j}, & \tau \le \Vert \vec{a}_{\bullet j} \Vert \\
        & 0, & \text{otherwise}
        \end{aligned}
        \right.\text{.}

    where :math:`\vec{x}_{\bullet j}` is the :math:`j`-th column of matrix :math:`\boldsymbol{X}`, 
    and :math:`\vec{a}_{\bullet j}` is the :math:`j`-th column of matrix :math:`\boldsymbol{A}`.

    :param A: Matrix :math:`\boldsymbol{A}`.
    :type A: np.ndarray
    :param tau: :math:`\tau`.
    :type tau: float
    :return: The solution to the optimization problem.
    :rtype: np.ndarray
    """
    norms = np.sqrt(np.sum(A**2, axis=0)).reshape(1, -1)
    return np.where(norms > tau, ((norms - tau) / norms) * A, 0.)


def normalize(D: np.ndarray) -> np.ndarray:
    return D / np.sum(D, axis=1, keepdims=True)


def proj(D: np.ndarray) -> np.ndarray:
    r"""This approach is proposed in paper :cite:`2016:condat`.

    :param D: Matrix :math:`\boldsymbol{D}`.
    :type D: np.ndarray
    :return: The projection onto the probability simplex.
    :rtype: np.ndarray
    """
    X = -np.sort(-D, axis=1)
    Xtmp = (np.cumsum(X, axis=1) - 1) / np.arange(1, D.shape[1] + 1)
    rho = np.sum(X > Xtmp, axis=1) - 1
    theta = Xtmp[np.arange(D.shape[0]), rho]
    return np.maximum(D - theta[:, np.newaxis], 0)


def locass_proj(D, k):
    a = D * k
    b = np.floor(a).astype(np.int64)
    remainder = a - b
    deficit = k - np.sum(b, axis=1)
    rank = np.argsort(np.argsort(-remainder, axis=1), axis=1)
    bonus = (rank < deficit[:, None]).astype(np.int64)
    return (b + bonus) / k


def softmax(D: np.ndarray) -> np.ndarray:
    from scipy.special import softmax as scipy_softmax
    return scipy_softmax(D, axis=1)


def shannon_entropy(D: np.ndarray):
    return -np.sum(D * np.log(D + EPS))


def estimate_alpha(D: np.ndarray, max_iterations: int = 100, convergence_criterion=1e-7):
    from scipy.special import digamma, polygamma

    def inv_digamma(y: np.ndarray) -> np.ndarray:
        x = np.where(y >= -2.22, np.exp(y) + .5, -1. / (y + np.euler_gamma))
        for _ in range(5):
            x = x - (digamma(x) - y) / polygamma(1, x)
        return x

    log_D_bar = np.mean(np.log(D + EPS), axis=0)
    alpha = np.ones(D.shape[1], dtype=np.float64)

    for _ in range(max_iterations):
        alpha0 = np.sum(alpha)
        y = digamma(alpha0) + log_D_bar
        alpha_new = inv_digamma(y)

        if np.max(np.abs(alpha_new - alpha)) < convergence_criterion:
            return alpha_new
        alpha = alpha_new

    return alpha


def binaryzation(D: np.ndarray, method='threshold', param: any = None) -> np.ndarray:
    r"""Transform label distribution matrix to logical label matrix.

    :param D: Label distribution matrix (shape: :math:`[n,\, l]`).
    :type D: np.ndarray
    :param method: Type of binaryzation method, defaults to 'threshold'. The options are 'threshold' and 'topk', which can refer to:

        .. bibliography:: bib/ldl/references.bib
            :filter: False
            :labelprefix: BIN-
            :keyprefix: bin-

            2024:kou

    :type method: {'threshold', 'topk'}, optional
    :param param: Parameter of binaryzation method, defaults to None. If None, the default value is .5 for 'threshold' and :math:`\lfloor l / 2 \rfloor` for 'topk'.
    :type param: any, optional
    :return: Logical label matrix (shape: :math:`[n,\, l]`).
    :rtype: np.ndarray
    """
    r = np.argsort(np.argsort(D))

    if method == 'threshold':
        if param is None:
            param = .5
        elif not isinstance(param, float) or param < 0. or param >= 1.:
            raise ValueError("Invalid param, when method is 'threshold', "
                             "param should be a float in the range [0, 1).")
        b = np.sort(D.T, axis=0)[::-1]
        cs = np.cumsum(b, axis=0)
        m = np.argmax(cs >= param, axis=0)
        return np.where(r >= D.shape[1] - m.reshape(-1, 1) - 1, 1, 0)

    elif method == 'topk':
        if param is None:
            param = D.shape[1] // 2
        elif not isinstance(param, int) or param < 1 or param >= D.shape[1]:
            raise ValueError("Invalid param, when method is 'topk', "
                             "param should be an integer in the range [1, number_of_labels).")
        return np.where(r >= D.shape[1] - param, 1, 0)

    else:
        raise ValueError("Invalid method, which should be 'threshold' or 'topk'.")


@_1d
def pairwise_euclidean(X, Y=None):

    @singledispatch
    def _pairwise(X, Y):
        import keras.ops as ops
        return ops.sqrt(ops.sum((X[:, None] - Y[None, :])**2, axis=2) + EPS)

    @_pairwise.register(np.ndarray)
    def _(X: np.ndarray, Y: np.ndarray):
        return np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis]) ** 2, axis=2) + EPS)

    Y = X if Y is None else Y
    return _pairwise(X, Y)


@_1d
def pairwise_cosine(X, Y=None, mode: str = 'similarity'):

    @singledispatch
    def _pairwise(X, Y):
        import keras.ops as ops
        X_norm = X / ops.sqrt(ops.sum(X**2, axis=1, keepdims=True))
        Y_norm = Y / ops.sqrt(ops.sum(Y**2, axis=1, keepdims=True))
        similarity = ops.matmul(X_norm, ops.transpose(Y_norm))
        return similarity

    @_pairwise.register(np.ndarray)
    def _(X: np.ndarray, Y: np.ndarray):
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        similarity = np.dot(X_norm, np.transpose(Y_norm))
        return similarity

    Y = X if Y is None else Y
    similarity = _pairwise(X, Y)
    return 1 - similarity if mode == 'distance' else similarity


@_1d
def pairwise_pearsonr(X, Y=None):

    @singledispatch
    def _pairwise(X, Y):
        import keras.ops as ops
        Y = X if Y is None else Y
        X_centered = X - ops.mean(X, axis=1, keepdims=True)
        Y_centered = Y - ops.mean(Y, axis=1, keepdims=True)
        cov = ops.matmul(X_centered, ops.transpose(Y_centered))
        X_std = ops.sqrt(ops.sum(X_centered**2, axis=1, keepdims=True))
        Y_std = ops.sqrt(ops.sum(Y_centered**2, axis=1, keepdims=True))
        return cov / ops.matmul(X_std, ops.transpose(Y_std))

    @_pairwise.register(np.ndarray)
    def _(X: np.ndarray, Y: Optional[np.ndarray]):
        return np.corrcoef(X) if Y is None else np.corrcoef(X, Y)[:X.shape[0], X.shape[0]:]

    return _pairwise(X, Y)


def csr2sparse(A):
    import tensorflow as tf
    coo = A.tocoo()
    indices = np.asmatrix([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(
        tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    )


def kernel(X, Y=None, gamma: Optional[float] = None):
    from sklearn.metrics.pairwise import rbf_kernel
    Y = X if Y is None else Y
    if gamma is None:
        gamma = 1. / (2. * np.mean(pairwise_euclidean(X, Y)) ** 2)
    return rbf_kernel(X, Y, gamma=gamma)


def non_diagonal(X):

    @singledispatch
    def _non_diagonal(X):
        import keras.ops as ops
        return  X * (1 - ops.eye(ops.shape(X)[0], dtype=X.dtype))

    @_non_diagonal.register(np.ndarray)
    def _(X: np.ndarray):
        return X - np.diag(np.diag(X))

    return _non_diagonal(X)
