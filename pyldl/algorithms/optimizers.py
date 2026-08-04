import keras
import keras.ops as ops


class RProp(keras.optimizers.Optimizer):

    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp, self).__init__(name='rprop', learning_rate=init_alpha, **kwargs)
        self._init_alpha = init_alpha
        self._scale_up = scale_up
        self._scale_down = scale_down
        self._min_alpha = min_alpha
        self._max_alpha = max_alpha

    def build(self, variables):
        if self.built:
            return
        super().build(variables)
        self._alphas = []
        self._old_grads = []
        self._prev_weight_deltas = []
        for variable in variables:
            self._alphas.append(
                self.add_variable(
                    shape=variable.shape,
                    initializer=keras.initializers.Constant(self._init_alpha),
                    dtype=variable.dtype,
                    name="alpha",
                )
            )
            self._old_grads.append(
                self.add_variable(
                    shape=variable.shape,
                    initializer="zeros",
                    dtype=variable.dtype,
                    name="old_grad",
                )
            )
            self._prev_weight_deltas.append(
                self.add_variable(
                    shape=variable.shape,
                    initializer="zeros",
                    dtype=variable.dtype,
                    name="prev_weight_delta",
                )
            )

    def update_step(self, gradient, variable, _):
        idx = self._get_variable_index(variable)
        alpha = self._alphas[idx]
        old_grad = self._old_grads[idx]
        prev_weight_delta = self._prev_weight_deltas[idx]
        grad_prod = gradient * old_grad

        new_alpha = ops.where(
            grad_prod > 0.,
            ops.minimum(alpha * self._scale_up, self._max_alpha),
            ops.where(grad_prod < 0., ops.maximum(alpha * self._scale_down, self._min_alpha), alpha)
        )

        new_delta = ops.where(gradient > 0., -new_alpha,
                              ops.where(gradient < 0., new_alpha, ops.zeros_like(new_alpha)))

        weight_delta = ops.where(grad_prod < 0., -prev_weight_delta, new_delta)
        gradient = ops.where(grad_prod < 0., ops.zeros_like(gradient), gradient)

        self.assign(variable, variable + weight_delta)
        alpha.assign(new_alpha)
        old_grad.assign(gradient)
        prev_weight_delta.assign(weight_delta)

    def get_config(self):
        config = {
            'init_alpha': self._init_alpha,
            'scale_up': self._scale_up,
            'scale_down': self._scale_down,
            'min_alpha': self._min_alpha,
            'max_alpha': self._max_alpha,
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
