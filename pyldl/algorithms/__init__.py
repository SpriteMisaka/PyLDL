import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from ._problem_transformation import _PT, PT_Bayes, PT_SVM, LDSVR
from ._algorithm_adaptation import AA_BP, AA_KNN, CPNN, BCPNN, ACPNN, LDLF, Duo_LDL
from ._specialized_algorithms import _SA, SA_BFGS, SA_IIS, LALOT

from ._incomplete import IncomLDL, WInLDL
from ._classifier import LDL4C, LDL_HR, LDLM
from ._ensemble import RG4LDL, DF_LDL, StructRF, AdaBoostLDL

from ._ldllc import LDLLC
from ._ldlsf import LDLSF
from ._ldl_lclr import LDL_LCLR
from ._ldl_scl import LDL_SCL
from ._ldl_lrr import LDL_LRR
from ._ldl_dpa import LDL_DPA
from ._lrldl import _LRLDL, TLRLDL, TKLRLDL
from ._ldl_hvlc import LDL_HVLC
from ._rknn_ldl import RKNN_LDL
from ._s_ldl import _S_LDL, S_LRR, S_SCL, S_KLD, S_QFD2, S_CJS, Shallow_S_LDL
from ._delta_ldl import Delta_LDL

from ._ssg_ldl import SSG_LDL

from ._label_enhancement import FCM, KM, LP, ML, GLLE, LEVI, LIBLE, ConLE

from ._ldl_da import LDL_DA


_ldl__ = [
# -------------------- 2025 --------------------
"RG4LDL", "RKNN_LDL", "_S_LDL", "S_LRR", "S_SCL", "S_KLD", "S_CJS", "S_QFD2", "Shallow_S_LDL", "Delta_LDL",
# -------------------- 2024 --------------------
"LDL_HVLC", "_LRLDL", "TKLRLDL", "TLRLDL", "LDL_DPA",
# -------------------- 2023 --------------------
"LDL_LRR",
# -------------------- 2021 --------------------
"DF_LDL", "LDL_SCL", "Duo_LDL",
# -------------------- 2020 --------------------
"AdaBoostLDL",
# -------------------- 2019 --------------------
"LDL_LCLR", "LDLSF",
# -------------------- 2018 --------------------
"LDLLC", "LALOT", "StructRF",
# -------------------- 2017 --------------------
"BCPNN", "ACPNN", "LDLF",
# -------------------- 2016 --------------------
"_SA", "SA_BFGS", "SA_IIS", "AA_KNN", "AA_BP", "_PT", "PT_Bayes", "PT_SVM",
# -------------------- 2015 --------------------
"LDSVR",
# -------------------- 2013 --------------------
"CPNN",
]

_le__ = [
# -------------------- 2023 --------------------
"LIBLE", "ConLE",
# -------------------- 2020 --------------------
"LEVI",
# -------------------- 2019 --------------------
"GLLE", "ML", "LP", "KM", "FCM",
]

_incomldl__ = [
# -------------------- 2024 --------------------
"WInLDL",
# -------------------- 2017 --------------------
"IncomLDL",
]

_ldl4c__ = [
# -------------------- 2021 --------------------
"LDLM", "LDL_HR",
# -------------------- 2019 --------------------
"LDL4C",
]

_ssg_ldl__ = [
# -------------------- 2021 --------------------
"SSG_LDL",
]

_ldl_da__ = [
# -------------------- 2025 --------------------
"LDL_DA",
]

__all__ = _ldl__ + _le__ + _incomldl__ + _ldl4c__ + _ssg_ldl__ + _ldl_da__
