import numpy as np

import keras
import keras.ops as ops

from pyldl.algorithms.utils import _clip


EPS = np.finfo(np.float64).eps


def cad(D, D_pred):
    """This loss function is proposed in paper :cite:`2023:wen`.
    """
    def _cad(D, D_pred):
        return ops.mean(ops.abs(
            ops.cumsum(D, axis=1) - ops.cumsum(D_pred, axis=1)
        ), axis=1)
    return ops.sum(ops.stack([
        _cad(D[:, :i], D_pred[:, :i])
        for i in range(1, D.shape[1] + 1)
    ]))


def qfd2(D, D_pred):
    """This loss function is proposed in paper :cite:`2023:wen`.
    """
    Q = D - D_pred
    j = ops.reshape(ops.arange(D.shape[1]), [D.shape[1], 1])
    k = ops.reshape(ops.arange(D.shape[1]), [1, D.shape[1]])
    A = ops.cast(1 - ops.abs(j - k) / (D.shape[1] - 1), dtype="float32")
    return ops.mean(ops.sum(ops.matmul(Q, A) * Q, axis=1))


@_clip
def cjs(D, D_pred):
    """This loss function is proposed in paper :cite:`2023:wen`.
    """
    def _cjs(D, D_pred):
        m = 0.5 * (D + D_pred)
        js = 0.5 * (keras.losses.kl_divergence(D, m) + keras.losses.kl_divergence(D_pred, m))
        return ops.mean(js)
    return ops.sum(ops.stack([
        _cjs(D[:, :i], D_pred[:, :i])
        for i in range(1, D.shape[1] + 1)
    ]))


def unimodal_loss(y, D_pred):
    """This loss function is proposed in paper :cite:`2022:li`.
    """
    diff = (D_pred - ops.roll(D_pred, shift=-1, axis=1))[:, :-1]
    seq = ops.arange(1, D_pred.shape[1], dtype=y.dtype)
    sgn = ops.cast(
        ops.sign(seq[None, :] - ops.reshape(y, (-1, 1))),
        dtype="float32"
    )
    return ops.mean(ops.sum(ops.maximum(0, -diff * sgn), axis=1))


@_clip
def concentrated_loss(y, D_pred):
    """This loss function is proposed in paper :cite:`2022:li`.
    """
    seq = ops.arange(1, D_pred.shape[1] + 1, dtype="float32")
    y_pred = ops.sum(D_pred * seq[None, :], axis=1)
    v = ops.sum(D_pred * ((seq[None, :] - y_pred[:, None]) ** 2), axis=1)
    return ops.mean(.5 * ops.log(v + EPS) + (y_pred - y) ** 2 / (2 * v + EPS) + .5 * np.log(2 * np.pi))
