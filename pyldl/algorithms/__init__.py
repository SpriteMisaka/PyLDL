from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._problem_transformation import *
    from ._algorithm_adaptation import *
    from ._specialized_algorithms import *

    from ._incomplete import *
    from ._classifier import *
    from ._ensemble import *

    from ._bp import *
    from ._cpnn import *
    from ._duo_ldl import *
    from ._ldlf import *
    from ._ldllc import *
    from ._ldlsf import *
    from ._ldl_lclr import *
    from ._ldl_scl import *
    from ._ldl_lrr import *
    from ._ldl_dpa import *
    from ._lrldl import *
    from ._ldl_hvlc import *
    from ._rknn_ldl import *
    from ._s_ldl import *
    from ._delta_ldl import *
    from ._snefy_ldl import *
    from ._ldl_dvs import *
    from ._ldl_dpm import *

    from ._ssg_ldl import *

    from ._label_enhancement import *

    from ._ldl_da import *


_LDL_MODULE_MAP = {
    # -------------------- 2026 --------------------
    "LDL_DVS": "._ldl_dvs",
    "LDL_DPM": "._ldl_dpm",
    # -------------------- 2025 --------------------
    "RG4LDL": "._ensemble",
    "RKNN_LDL": "._rknn_ldl",
    "SNEFY_LDL": "._snefy_ldl",
    **dict.fromkeys(
        [
            "_S_LDL",
            "S_LRR",
            "S_SCL",
            "S_KLD",
            "S_CJS",
            "S_QFD2",
            "Shallow_S_LDL",
        ],
        "._s_ldl"
    ),
    "Delta_LDL": "._delta_ldl",
# -------------------- 2024 --------------------
    "LDL_HVLC": "._ldl_hvlc",
    **dict.fromkeys(
        [
            "_LRLDL",
            "TKLRLDL",
            "TLRLDL",
        ],
        "._lrldl"
    ),
    "LDL_DPA": "._ldl_dpa",
    # -------------------- 2023 --------------------
    "LDL_LRR": "._ldl_lrr",
    # -------------------- 2021 --------------------
    "DF_LDL": "._ensemble",
    "LDL_SCL": "._ldl_scl",
    "Duo_LDL": "._duo_ldl",
    "BD_LDL": "._algorithm_adaptation",
    # -------------------- 2019 --------------------
    "LDL_LCLR": "._ldl_lclr",
    "LDLSF": "._ldlsf",
    # -------------------- 2018 --------------------
    "LDLLC": "._ldllc",
    "LALOT": "._specialized_algorithms",
    "StructRF": "._ensemble",
    # -------------------- 2017 --------------------
    "BCPNN": "._cpnn",
    "ACPNN": "._cpnn",
    "LDLF": "._ldlf",
    # -------------------- 2016 --------------------
    "LDLogitBoost": "._ensemble",
    **dict.fromkeys(
        [
            "_SA",
            "SA_BFGS",
            "SA_IIS",
        ],
        "._specialized_algorithms"
    ),
    "AA_BP": "._bp",
    "AA_KNN": "._algorithm_adaptation",
    **dict.fromkeys(
        [
            "_PT",
            "PT_Bayes",
            "PT_SVM",
        ],
        "._problem_transformation"
    ),
    # -------------------- 2015 --------------------
    "LDSVR": "._problem_transformation",
    # -------------------- 2013 --------------------
    "CPNN": "._cpnn",
}

_LE_MODULE_MAP = {
    # -------------------- 2023 --------------------
    **dict.fromkeys(
        [
            "LIBLE",
            "ConLE",
        ],
        "._label_enhancement"
    ),
    # -------------------- 2020 --------------------
    "LEVI": "._label_enhancement",
    # -------------------- 2019 --------------------
    **dict.fromkeys(
        [
            "GLLE",
            "ML",
            "LP",
            "KM",
            "FCM",
        ],
        "._label_enhancement"
    ),
}

_INCOMLDL_MODULE_MAP = {
    # -------------------- 2024 --------------------
    "WInLDL": "._incomplete",
    # -------------------- 2017 --------------------
    "IncomLDL": "._incomplete",
}

_LDL4C_MODULE_MAP = {
    # -------------------- 2021 --------------------
    **dict.fromkeys(
        [
            "LDLM",
            "LDL_HR",
            "LDL4C",
        ],
        "._classifier"
    ),
}

_SSG_LDL_MODULE_MAP = {
    # -------------------- 2021 --------------------
    "SSG_LDL": "._ssg_ldl"
}

_LDL_DA_MODULE_MAP = {
    # -------------------- 2025 --------------------
    "LDL_DA": "._ldl_da"
}

_GLD_MODULE_MAP = {
    # -------------------- 2026 --------------------
    "GLD_SVR": "._problem_transformation",
    "GLD_KNN": "._algorithm_adaptation",  
    "GLD_BFGS": "._specialized_algorithms",
}

_MODULE_MAP = {
    **_LDL_MODULE_MAP,
    **_LE_MODULE_MAP,
    **_INCOMLDL_MODULE_MAP,
    **_LDL4C_MODULE_MAP,
    **_SSG_LDL_MODULE_MAP,
    **_LDL_DA_MODULE_MAP,
    **_GLD_MODULE_MAP,
}


_ldl__ = list(_LDL_MODULE_MAP)
_le__ = list(_LE_MODULE_MAP)
_incomldl__ = list(_INCOMLDL_MODULE_MAP)
_ldl4c__ = list(_LDL4C_MODULE_MAP)
_ssg_ldl__ = list(_SSG_LDL_MODULE_MAP)
_ldl_da__ = list(_LDL_DA_MODULE_MAP)
_gld__ = list(_GLD_MODULE_MAP)


__all__ = _ldl__ + _le__ + _incomldl__ + _ldl4c__ + _ssg_ldl__ + _ldl_da__ + _gld__


def __getattr__(name):

    if name not in _MODULE_MAP:
        raise AttributeError(name)

    import importlib

    module = importlib.import_module(
        _MODULE_MAP[name],
        package=__name__
    )

    obj = getattr(module, name)

    globals()[name] = obj

    return obj
