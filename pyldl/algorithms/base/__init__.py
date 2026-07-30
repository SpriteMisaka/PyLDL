from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .shallow import *
    from .deep import *


_MODULE_MAP = {
    **dict.fromkeys(
        [
            "Base",
            "BaseLDL",
            "BaseLE",
            "BaseGLD",
            "BaseADMM",
            "BaseIncomLDL",
            "BaseLDLClassifier",
            "BaseEnsemble",
        ],
        ".shallow",
    ),

    **dict.fromkeys(
        [
            "BaseDeep",
            "BaseDeepLDL",
            "BaseDeepLE",
            "BaseGD",
            "BaseAdam",
            "BaseBFGS",
            "BaseDeepLDLClassifier",
        ],
        ".deep",
    ),
}

__all__ = list(_MODULE_MAP)


from ._lazy import lazy_module
__getattr__ = lazy_module(_MODULE_MAP, __name__)
