import keras

from pyldl.algorithms.base.deep import BaseDeepLDL, BaseGD


@keras.saving.register_keras_serializable()
class AA_BP(BaseGD, BaseDeepLDL):
    """:class:`AA-BP <pyldl.algorithms.AA_BP>` is proposed in paper :cite:`2016:geng`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
