import keras
import tensorflow as tf


class _CAD():

    @staticmethod
    @tf.function
    def loss_function(D, D_pred):
        def _CAD(D, D_pred):
            return tf.reduce_mean(tf.abs(
                tf.cumsum(D, axis=1) - tf.cumsum(D_pred, axis=1)
            ), axis=1)
        return tf.math.reduce_sum(
            tf.map_fn(lambda i: _CAD(D[:, :i], D_pred[:, :i]),
                      tf.range(1, D.shape[1] + 1),
                      fn_output_signature=tf.float32)
        )


class _QFD2():

    @staticmethod
    @tf.function
    def loss_function(D, D_pred):
        Q = D - D_pred
        j = tf.reshape(tf.range(D.shape[1]), [D.shape[1], 1])
        k = tf.reshape(tf.range(D.shape[1]), [1, D.shape[1]])
        A = tf.cast(1 - tf.abs(j - k) / (D.shape[1] - 1), dtype=tf.float32)
        return tf.math.reduce_mean(
            tf.linalg.diag_part(tf.matmul(tf.matmul(Q, A), tf.transpose(Q)))
        )


class _CJS():

    @staticmethod
    @tf.function
    def loss_function(D, D_pred):
        def _CJS(D, D_pred):
            m = 0.5 * (D + D_pred)
            js = 0.5 * (keras.losses.kl_divergence(D, m) + keras.losses.kl_divergence(D_pred, m))
            return tf.reduce_mean(js)
        return tf.math.reduce_sum(
            tf.map_fn(lambda i: _CJS(D[:, :i], D_pred[:, :i]),
                      tf.range(1, D.shape[1] + 1),
                      fn_output_signature=tf.float32)
        )
