import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

import keras
import keras.ops as ops

from pyldl.algorithms.base import BaseDeepLDL, BaseAdam


EPS = np.finfo(np.float32).eps


@keras.saving.register_keras_serializable()
class LDL_SCL(BaseAdam, BaseDeepLDL):
    """:class:`LDL-SCL <pyldl.algorithms.LDL_SCL>` is proposed in paper :cite:`2018:zheng`. 
    SCL refers to (exploiting) *sample correlations locally*.

    :term:`Adam` is used as the optimizer.

    See also:

    .. bibliography:: bib/ldl/references.bib
        :filter: False
        :labelprefix: LDL-SCL-
        :keyprefix: ldl-scl-

        2021:jia
    """

    def __init__(self, n_clusters=5, alpha=1e-3, beta=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta

    def _get_default_model(self):
        return self.get_2layer_model(self._n_features, self._n_outputs)

    def _before_train(self):
        self._P = ops.convert_to_tensor(
            KMeans(n_clusters=self.n_clusters).fit(self._D).cluster_centers_,
            dtype="float32"
        )
        self._C = self.add_weight(
            name='C', shape=(self._n_samples, self.n_clusters),
            initializer=keras.initializers.Constant(EPS), trainable=True
        )
        self._W = self.add_weight(
            name='W', shape=(self.n_clusters, self._n_outputs),
            initializer=keras.initializers.RandomNormal(), trainable=True
        )

    @staticmethod
    def scl_loss(D_pred, P, C):
        corr = ops.mean(C * keras.losses.mean_squared_error(
            ops.expand_dims(D_pred, 1), ops.expand_dims(P, 0)
        ))
        barr = ops.mean(1 / C)
        return corr, barr

    def _loss(self, X, D, start, end):
        D_pred = keras.activations.softmax(self._model(X) + ops.matmul(self._C[start:end], self._W))
        kl = ops.mean(keras.losses.kl_divergence(D, D_pred))
        corr, barr = self.scl_loss(D_pred, self._P, self._C[start:end])
        return kl + self.alpha * corr + self.beta * barr

    @staticmethod
    def construct_C(X, old_X, old_C):
        C = np.zeros((X.shape[0], old_C.shape[1]))
        for i in range(old_C.shape[1]):
            lr = LinearRegression()
            lr.fit(old_X, old_C.numpy()[:, i].reshape(-1))
            C[:, i] = lr.predict(X).reshape(1, -1)
        return ops.convert_to_tensor(C, dtype="float32")

    _serialize_objects = ['_model', '_X', '_C', '_W']
    def predict(self, X):
        C = self.construct_C(X, self._X, self._C)
        return self._to_numpy(
            keras.activations.softmax(self._model(X) + ops.matmul(C, self._W))
        )
