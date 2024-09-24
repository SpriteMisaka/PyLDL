from typing import Union, Optional

import numpy as np

import keras
import tensorflow as tf

from keras import backend as K


EPS = np.finfo(np.float64).eps

DEFAULT_METRICS = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]


def _clip(func):
    def _wrapper(y, y_pred):
        y = np.clip(y, EPS, 1)
        y_pred = np.clip(y_pred, EPS, 1)
        return func(y, y_pred)
    return _wrapper


def _reduction(func):
    def _wrapper(*args, reduction=np.average):
        results = func(*args)
        return reduction(results) if reduction is not None else results
    return _wrapper


@_reduction
@_clip
def kl_divergence(y, y_pred):
    return np.sum(y * (np.log(y) - np.log(y_pred)), 1)


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


def soft_thresholding(A: np.ndarray, tau: float) -> np.ndarray:
    """Soft thresholding operation.
    It is defined as :math:`\\text{soft}(\\boldsymbol{A}, \\tau) = \\text{sgn}(\\boldsymbol{A}) \\odot \\max\\lbrace \\lvert \\boldsymbol{A} \\rvert - \\tau, 0 \\rbrace`, 
    where :math:`\\odot` denotes element-wise multiplication.

    :param A: Matrix :math:`\\boldsymbol{A}`.
    :type A: np.ndarray
    :param tau: :math:`\\tau`.
    :type tau: float
    :return: The result of soft thresholding operation.
    :rtype: np.ndarray
    """
    return np.sign(A) * np.maximum(np.abs(A) - tau, 0.)


def svt(A: np.ndarray, tau: float) -> np.ndarray:
    """Singular value thresholding (SVT) is proposed in paper :cite:`2010:cai`.

    The solution to the optimization problem 
    :math:`\\mathop{\\arg\\min}_{\\boldsymbol{X}} \\Vert \\boldsymbol{X} - \\boldsymbol{A} \\Vert_\\text{F}^2 + \\tau \\Vert \\boldsymbol{X} \\Vert_{\\ast}` 
    is given by :math:`\\boldsymbol{U} \\max \\lbrace \\boldsymbol{\\Sigma} - \\tau, 0 \\rbrace \\boldsymbol{V}^\\text{T}`, 
    where :math:`\\boldsymbol{A} = \\boldsymbol{U} \\boldsymbol{\\Sigma} \\boldsymbol{V}^\\text{T}` is the singular value decomposition of matrix :math:`\\boldsymbol{A}`.

    :param A: Matrix :math:`\\boldsymbol{A}`.
    :type A: np.ndarray
    :param tau: :math:`\\tau`.
    :type tau: float
    :return: The solution to the optimization problem.
    :rtype: np.ndarray
    """
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0.)
    return U @ np.diag(S_thresh) @ VT


def solvel21(A: np.ndarray, tau: float) -> np.ndarray:
    """This approach is proposed in paper :cite:`2014:chen`.

    The solution to the optimization problem 
    :math:`\\mathop{\\arg\\min}_{\\boldsymbol{X}} \\Vert \\boldsymbol{X} - \\boldsymbol{A} \\Vert_\\text{F}^2 + \\tau \\Vert \\boldsymbol{X} \\Vert_{2,\\,1}` 
    is given by the following formula: 

    .. math::

        \\vec{x}_{\\bullet j}^{\\ast} = \\left\\{
        \\begin{aligned}
        & \\frac{\\Vert \\vec{a}_{\\bullet j} \\Vert - \\tau}{\\Vert \\vec{a}_{\\bullet j} \\Vert} \\vec{a}_{\\bullet j}, & \\tau \\le \\Vert \\vec{a}_{\\bullet j} \\Vert \\\\
        & 0, & \\text{otherwise}
        \\end{aligned}
        \\right.

    where :math:`\\vec{x}_{\\bullet j}` is the :math:`j`-th column of matrix :math:`\\boldsymbol{X}`, 
    and :math:`\\vec{a}_{\\bullet j}` is the :math:`j`-th column of matrix :math:`\\boldsymbol{A}`.

    :param A: Matrix :math:`\\boldsymbol{A}`.
    :type A: np.ndarray
    :param tau: :math:`\\tau`.
    :type tau: float
    :return: The solution to the optimization problem.
    :rtype: np.ndarray
    """
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    return np.where(norms > tau, ((norms - tau) / norms) * A, 0.)


def proj(Y: np.ndarray) -> np.ndarray:
    """This approach is proposed in paper :cite:`2016:condat`.

    :param Y: Matrix :math:`\\boldsymbol{Y}`.
    :type Y: np.ndarray
    :return: The projection onto the probability simplex.
    :rtype: np.ndarray
    """
    X = -np.sort(-Y, axis=1)
    Xtmp = (np.cumsum(X, axis=1) - 1) / np.arange(1, Y.shape[1] + 1)
    rho = np.sum(X > Xtmp, axis=1) - 1
    theta = Xtmp[np.arange(Y.shape[0]), rho]
    return np.maximum(Y - theta[:, np.newaxis], 0)


def binaryzation(y: np.ndarray, method='threshold', param: any = None) -> np.ndarray:
    """Transform label distribution matrix to logical label matrix.

    :param y: Label distribution matrix (shape: :math:`[n,\\, l]`).
    :type y: np.ndarray
    :param method: Type of binaryzation method, defaults to 'threshold'. The options are 'threshold' and 'topk', which can refer to:

        .. bibliography:: ldl_references.bib
            :filter: False
            :labelprefix: BIN-
            :keyprefix: bin-

            2024:kou

    :type method: {'threshold', 'topk'}, optional
    :param param: Parameter of binaryzation method, defaults to None. If None, the default value is .5 for 'threshold' and :math:`\\lfloor l / 2 \\rfloor` for 'topk'.
    :type param: any, optional
    :return: Logical label matrix (shape: :math:`[n,\\, l]`).
    :rtype: np.ndarray
    """
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


def pairwise_euclidean(X: Union[np.ndarray, tf.Tensor],
                       Y: Optional[Union[np.ndarray, tf.Tensor]] = None) -> Union[np.ndarray, tf.Tensor]:
    """Pairwise Euclidean distance.

    :param X: Matrix :math:`\\boldsymbol{X}` (shape: :math:`[m_X,\\, n_X]`).
    :type X: Union[np.ndarray, tf.Tensor]
    :param Y: Matrix :math:`\\boldsymbol{Y}` (shape: :math:`[m_Y,\\, n_Y]`), if None, :math:`\\boldsymbol{Y} = \\boldsymbol{X}`, defaults to None.
    :type Y: Union[np.ndarray, tf.Tensor], optional
    :return: Pairwise Euclidean distance (shape: :math:`[m_X,\\, m_Y]`).
    :rtype: Union[np.ndarray, tf.Tensor]
    """
    Y = X if Y is None else Y
    if isinstance(X, np.ndarray):
        return np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis]) ** 2, axis=2))
    elif isinstance(X, tf.Tensor):
        return tf.sqrt(tf.reduce_sum((X[:, tf.newaxis] - Y[tf.newaxis]) ** 2, axis=2))
    else:
        raise TypeError("Input must be either a tf.Tensor or a np.ndarray")


class RProp(keras.optimizers.Optimizer):

    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp, self).__init__(name='rprop', **kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def apply_gradients(self, grads_and_vars):
        grads, trainable_variables = zip(*grads_and_vars)
        self.get_updates(trainable_variables, grads)

    def get_updates(self, params, gradients):
        grads = gradients
        shapes = [K.int_shape(p) for p in params]
        alphas = [K.variable(np.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        self.updates = []

        for param, grad, old_grad, prev_weight_delta, alpha in zip(params, grads,
                                                                   old_grads, prev_weight_deltas,
                                                                   alphas):

            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0), K.maximum(alpha * self.scale_down, self.min_alpha), alpha)
            )

            new_delta = K.switch(K.greater(grad, 0),
                                 -new_alpha,
                                 K.switch(K.less(grad, 0),
                                          new_alpha,
                                          K.zeros_like(new_alpha)))

            weight_delta = K.switch(K.less(grad*old_grad, 0), -prev_weight_delta, new_delta)

            new_param = param + weight_delta

            grad = K.switch(K.less(grad*old_grad, 0), K.zeros_like(grad), grad)

            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(prev_weight_delta, weight_delta))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
