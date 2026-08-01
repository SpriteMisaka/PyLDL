import keras
import keras.ops as ops

import numpy as np

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


@keras.saving.register_keras_serializable()
class LDLF(BaseAdam, BaseDeepLDL):
    """:class:`LDLF <pyldl.algorithms.LDLF>` is proposed in paper :cite:`2017:shen`.

    :term:`Adam` is used as the optimizer.

    This algorithm employs deep neural decision forests. See also:

    .. bibliography:: bib/ldl/references.bib
        :filter: False
        :labelprefix: LDLF-
        :keyprefix: ldlf-

        2015:kontschieder
    """

    def __init__(self, n_estimators=5, n_depth=6, n_hidden=64, n_latent=64, **kwargs):
        super().__init__(n_hidden, n_latent, **kwargs)
        self._n_estimators = n_estimators
        self._n_depth = n_depth
        self._n_leaves = 2 ** n_depth

    def _get_default_model(self):
        return self.get_3layer_model(self._n_features, self._n_hidden, self._n_latent, output_activation='sigmoid')

    def _before_train(self):
        self._phi = [np.random.choice(
            np.arange(self._n_latent), size=self._n_leaves, replace=False
        ) for _ in range(self._n_estimators)]

        self._pi = [keras.Variable(
            initializer=keras.initializers.Constant(1 / self._n_outputs),
            shape=[self._n_leaves, self._n_outputs],
            dtype="float32", trainable=False,
        ) for _ in range(self._n_estimators)]

    def _loss(self, X, D, *_):
        loss = 0.
        self._pi_new = []
        for i in range(self._n_estimators):
            _mu = self._call(X, i)
            _prob = ops.matmul(_mu, self._pi[i])

            loss += ops.mean(keras.losses.kl_divergence(D, _prob))

            _D = ops.expand_dims(D, axis=1)
            _pi = ops.expand_dims(self._pi[i], axis=0)
            _mu = ops.expand_dims(_mu, axis=2)
            _prob = ops.clip(
                ops.expand_dims(_prob, axis=1), 1e-7, 1.)
            _new_pi = ops.multiply(ops.multiply(_D, _pi), _mu) / _prob
            _new_pi = ops.sum(_new_pi, axis=0)
            _new_pi = keras.activations.softmax(_new_pi)
            self._pi_new.append(_new_pi)

        loss /= self._n_estimators
        return loss

    def train_step(self, pair, batch, loss, start, end):
        super().train_step(pair, batch, loss, start, end)
        for i in range(self._n_estimators):
            self._pi[i].assign(ops.stop_gradient(self._pi_new[i]))

    _serialize_objects = ['_model', '_phi', '_pi']
    def _call(self, X, i):
        decisions = ops.take(self._model(X), self._phi[i], axis=1)
        decisions = ops.expand_dims(decisions, axis=2)
        decisions = ops.concatenate([decisions, 1 - decisions], axis=2)
        mu = ops.ones([X.shape[0], 1, 1])

        begin_idx = 1
        end_idx = 2

        for level in range(self._n_depth):
            mu = ops.reshape(mu, [X.shape[0], -1, 1])
            mu = ops.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = ops.reshape(mu, [X.shape[0], self._n_leaves])

        return mu

    def predict(self, X):
        from pyldl.algorithms.utils import proj
        res = np.zeros([X.shape[0], self._n_outputs], dtype=np.float32)
        for i in range(self._n_estimators):
            res += proj(self._to_numpy(
                ops.matmul(self._call(X, i), self._pi[i])
            ))
        return res / self._n_estimators

    @property
    def n_estimators(self):
        return self._n_estimators

    @property
    def n_depth(self):
        return self._n_depth

    @property
    def n_leaves(self):
        return self._n_leaves
