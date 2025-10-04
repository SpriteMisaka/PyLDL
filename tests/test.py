import unittest

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RepeatedKFold

from pyldl.utils import make_ldl
from pyldl.metrics import kl_divergence, worst_kl_divergence

from pyldl.algorithms import *


class Test(unittest.TestCase):

    def test(self):

        random_state = 42

        n_splits = 10
        n_repeats = 5

        methods = [SA_BFGS(), SA_IIS(), AA_KNN(), AA_BP(), PT_Bayes(), PT_SVM(),
                   LDSVR(), LALOT(), BD_LDL(), LDLF(), LDL_SCL(), LDL_LRR(), LDLSF()]

        X, D = make_ldl(random_state=random_state)
        worst_kl = worst_kl_divergence(D)
        print(f'Worst case: {worst_kl}')
        for method in methods:
            results = cross_val_score(method, X, D,
                cv=RepeatedKFold(
                    n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
                ),
                scoring=make_scorer(kl_divergence)
            )
            print(f'{method.__class__.__name__} on the artificial dataset: {results.mean()}')
            self.assertLess(results.mean(), worst_kl)
