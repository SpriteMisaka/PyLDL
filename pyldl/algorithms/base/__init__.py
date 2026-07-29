from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .shallow import *
    from .deep import *


__all__ = [

    # Shallow
    "Base", "BaseLDL", "BaseLE", "BaseGLD",
    "BaseADMM",
    "BaseIncomLDL", "BaseLDLClassifier",
    "BaseEnsemble",

    # Deep
    "BaseDeep", "BaseDeepLDL", "BaseDeepLE",
    "BaseGD", "BaseAdam", "BaseBFGS",
    "BaseDeepLDLClassifier",

]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)

    import importlib
    for module_name in (".shallow", ".deep"):
        try:
            module = importlib.import_module(
                module_name,
                package=__name__
            )
            return getattr(module, name)
        except AttributeError:
            pass

    raise AttributeError(name)
