import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from ._problem_transformation import PT_Bayes, PT_SVM, LDSVR
from ._algorithm_adaptation import AA_BP, AA_KNN, CAD, QFD2, CJS, CPNN, BCPNN, ACPNN
from ._specialized_algorithms import SA_BFGS, SA_IIS

from ._incomplete import IncomLDL, WInLDL
from ._classifier import LDL4C, LDL_HR, LDLM
from ._ensemble import DF_LDL, AdaBoostLDL

from ._ldlf import LDLF
from ._ldllc import LDLLC
from ._ldl_scl import LDL_SCL
from ._ldl_lrr import LDL_LRR
from ._ldl_dpa import LDL_DPA

from ._ssg_ldl import SSG_LDL

from ._label_enhancement import FCM, KM, LP, ML, GLLE, LEVI, LIBLE

from ._ldl_da import LDL_DA


__all__ = ["SA_BFGS", "SA_IIS", "AA_KNN", "AA_BP", "PT_Bayes", "PT_SVM",
           "CPNN", "BCPNN", "ACPNN", "LDSVR",
           "LDLF", "LDLLC", "LDL_SCL", "LDL_LRR", "LDL_DPA", "CAD", "QFD2", "CJS",
           "DF_LDL", "AdaBoostLDL",
           "LDL4C", "LDL_HR", "LDLM",
           "IncomLDL", "WInLDL",
           "SSG_LDL",
           "FCM", "KM", "LP", "ML", "GLLE",
           "LEVI", "LIBLE",
           "LDL_DA"]
