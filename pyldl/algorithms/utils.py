import numpy as np

import keras
import tensorflow as tf

from keras import backend as K

def pairwise_euclidean(X: tf.Tensor, Y: tf.Tensor = None) -> tf.Tensor:
    """Pairwise Euclidean distance.

    :param X: Matrix :math:`\\boldsymbol{X}` (shape: :math:`[m_X,\, n_X]`).
    :type X: tf.Tensor
    :param Y: Matrix :math:`\\boldsymbol{Y}` (shape: :math:`[m_Y,\, n_Y]`).
    :type Y: tf.Tensor
    :return: Pairwise Euclidean distance (shape: :math:`[m_X,\, m_Y]`).
    :rtype: tf.Tensor
    """
    Y = X if Y is None else Y
    X2 = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), [1, tf.shape(Y)[0]])
    Y2 = tf.tile(tf.reduce_sum(tf.square(Y), axis=1, keepdims=True), [1, tf.shape(X)[0]])
    XY = tf.matmul(X, tf.transpose(Y))
    return X2 + tf.transpose(Y2) - 2 * XY


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
