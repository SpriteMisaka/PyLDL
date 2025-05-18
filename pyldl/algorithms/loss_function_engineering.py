import numpy as np

import keras
import tensorflow as tf

from pyldl.algorithms.utils import _clip


EPS = np.finfo(np.float64).eps


@tf.function
def cad(D, D_pred):
    """This loss function is proposed in paper :cite:`2023:wen`.
    """
    @tf.function
    def _cad(D, D_pred):
        return tf.reduce_mean(tf.abs(
            tf.cumsum(D, axis=1) - tf.cumsum(D_pred, axis=1)
        ), axis=1)
    return tf.math.reduce_sum(
        tf.map_fn(lambda i: _cad(D[:, :i], D_pred[:, :i]),
                  tf.range(1, D.shape[1] + 1),
                  fn_output_signature=tf.float32)
    )


@tf.function
def qfd2(D, D_pred):
    """This loss function is proposed in paper :cite:`2023:wen`.
    """
    Q = D - D_pred
    j = tf.reshape(tf.range(D.shape[1]), [D.shape[1], 1])
    k = tf.reshape(tf.range(D.shape[1]), [1, D.shape[1]])
    A = tf.cast(1 - tf.abs(j - k) / (D.shape[1] - 1), dtype=tf.float32)
    return tf.math.reduce_mean(
        tf.linalg.diag_part(tf.matmul(tf.matmul(Q, A), tf.transpose(Q)))
    )


@tf.function
@_clip
def cjs(D, D_pred):
    """This loss function is proposed in paper :cite:`2023:wen`.
    """
    @tf.function
    def _cjs(D, D_pred):
        m = 0.5 * (D + D_pred)
        js = 0.5 * (keras.losses.kl_divergence(D, m) + keras.losses.kl_divergence(D_pred, m))
        return tf.reduce_mean(js)
    return tf.math.reduce_sum(
        tf.map_fn(lambda i: _cjs(D[:, :i], D_pred[:, :i]),
                  tf.range(1, D.shape[1] + 1),
                  fn_output_signature=tf.float32)
    )


@tf.function
def unimodal_loss(y, D_pred):
    """This loss function is proposed in paper :cite:`2022:li`.
    """
    diff = (D_pred - tf.roll(D_pred, shift=-1, axis=1))[:, :-1]
    seq = tf.range(1, tf.shape(D_pred)[1], dtype=y.dtype)
    sgn = tf.cast(
        tf.sign(seq[tf.newaxis, :] - tf.reshape(y, (-1, 1))),
        dtype=tf.float32
    )
    return tf.reduce_mean(tf.reduce_sum(tf.maximum(0, -diff * sgn), axis=1))


@tf.function
@_clip
def concentrated_loss(y, D_pred):
    """This loss function is proposed in paper :cite:`2022:li`.
    """
    seq = tf.range(1, tf.shape(D_pred)[1] + 1, dtype=tf.float32)
    y_pred = tf.reduce_sum(D_pred * seq[tf.newaxis, :], axis=1)
    v = tf.reduce_sum(D_pred * ((seq[tf.newaxis, :] - y_pred[:, tf.newaxis]) ** 2), axis=1)
    return tf.reduce_mean(.5 * tf.math.log(v + EPS) + (y_pred - y) ** 2 / (2 * v + EPS) + .5 * tf.math.log(2 * np.pi))
