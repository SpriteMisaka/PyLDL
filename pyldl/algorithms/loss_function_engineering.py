import keras
import tensorflow as tf


class _CAD():

    @staticmethod
    @tf.function
    def loss_function(y, y_pred):
        def _CAD(y, y_pred):
            return tf.reduce_mean(tf.abs(
                tf.cumsum(y, axis=1) - tf.cumsum(y_pred, axis=1)
            ), axis=1)
        return tf.math.reduce_sum(
            tf.map_fn(lambda i: _CAD(y[:, :i], y_pred[:, :i]),
                      tf.range(1, y.shape[1] + 1),
                      fn_output_signature=tf.float32)
        )


class _QFD2():

    @staticmethod
    @tf.function
    def loss_function(y, y_pred):
        Q = y - y_pred
        j = tf.reshape(tf.range(y.shape[1]), [y.shape[1], 1])
        k = tf.reshape(tf.range(y.shape[1]), [1, y.shape[1]])
        A = tf.cast(1 - tf.abs(j - k) / (y.shape[1] - 1), dtype=tf.float32)
        return tf.math.reduce_mean(
            tf.linalg.diag_part(tf.matmul(tf.matmul(Q, A), tf.transpose(Q)))
        )


class _CJS():

    @staticmethod
    @tf.function
    def loss_function(y, y_pred):
        def _CJS(y, y_pred):
            m = 0.5 * (y + y_pred)
            js = 0.5 * (keras.losses.kl_divergence(y, m) + keras.losses.kl_divergence(y_pred, m))
            return tf.reduce_mean(js)
        return tf.math.reduce_sum(
            tf.map_fn(lambda i: _CJS(y[:, :i], y_pred[:, :i]),
                      tf.range(1, y.shape[1] + 1),
                      fn_output_signature=tf.float32)
        )
