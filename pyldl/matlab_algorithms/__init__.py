import io
import os
import numpy as np
import matlab.engine

from pyldl.algorithms.base import BaseLDL

eng = matlab.engine.start_matlab()

def _set_arr(name, arr):
    eng.eval(f'global {name}', nargout=0)
    eng.workspace[f'{name}'] = matlab.double(arr.tolist())


def _get_arr(name):
    with io.StringIO() as target:
        eng.eval(f'tmp = {name}', nargout=0, stdout=target)
    return np.array(eng.workspace['tmp'])


class BaseMatlabLDL(BaseLDL):

    class MatlabLDLExec:

        def __enter__(self):
            eng.cd(f'{os.path.dirname(os.path.abspath(__file__))}', nargout=0)
            eng.eval(r'addpath("./LDLPackage_v1.2")', nargout=0)

        def __exit__(self, exc_type, exc_val, exc_tb):
            eng.eval('clear;', nargout=0)
            eng.eval('clc;', nargout=0)
            eng.eval(r'rmpath("./LDLPackage_v1.2")', nargout=0)
            eng.cd(r'..', nargout=0)

    def __init__(self, info_python=None, info_matlab=None, random_state=None):
        super().__init__(random_state)
        self._info_python, self._info_matlab = info_python, info_matlab
        exec(f"self._{self._info_python} = None")

    def fit(self, X, D):
        super().fit(X, D)

        with BaseMatlabLDL.MatlabLDLExec():
            _set_arr('features', X)
            _set_arr('labels', D)
            with io.StringIO() as target:
                exec(f"eng.{self.__class__.__name__}_fit(nargout=0, stdout=target)")
            if self._info_matlab == 'weights':
                self._W = np.array(eng.workspace['weights'])
            else:
                self._model = eng.workspace['model']

    def predict(self, X):
        with BaseMatlabLDL.MatlabLDLExec():
            _set_arr('features', X)
            if self._info_matlab == 'weights':
                exec(f"_set_arr('weights', self._{self._info_python})")
            else:
                eng.workspace['model'] = self._model
            with io.StringIO() as target:
                exec(f"eng.{self.__class__.__name__}_predict(nargout=0, stdout=target)")
            return np.array(eng.workspace['preDistribution'])


class _SA(BaseMatlabLDL):

    def __init__(self, random_state=None):
        super().__init__("W", "weights", random_state)


class SA_BFGS(_SA):
    pass


class SA_IIS(_SA):
    pass


class AA_KNN(BaseMatlabLDL):

    def __init__(self, random_state=None):
        BaseLDL.__init__(self, random_state)

    def fit(self, X, D):
        BaseLDL.fit(self, X, D)

    def predict(self, X):
        with BaseMatlabLDL.MatlabLDLExec():
            _set_arr('features', self._X)
            _set_arr('labels', self._D)
            _set_arr('testFeatures', X)
            with io.StringIO() as target:
                eng.AA_KNN(nargout=0, stdout=target)
            return np.array(eng.workspace['preDistribution'])


class AA_BP(BaseMatlabLDL):

    @staticmethod
    def _get_weights(name):
        size = _get_arr(f"size(net.{name})")[0].astype(np.int32)
        weights = [None for _ in range(size[0] * size[1])]
        for i in range(size[0] * size[1]):
            w = _get_arr(f"net.{name}" + r"{" + str(i // size[1] + 1) + r", " + str(i % size[1] + 1) + r"}").tolist()
            weights[i] = matlab.double(w)
        return weights

    @staticmethod
    def _set_weights(name, weights):
        eng.workspace[f'{name}'] = weights

    def _save_net(self):
        self._IW = AA_BP._get_weights("IW")
        self._LW = AA_BP._get_weights("LW")
        self._b = AA_BP._get_weights("b")

    def _load_net(self):
        AA_BP._set_weights("IW", self._IW)
        AA_BP._set_weights("LW", self._LW)
        AA_BP._set_weights("b", self._b)
        

    def __init__(self, random_state=None):
        BaseLDL.__init__(self, random_state)
    
    def fit(self, X, y):
        BaseLDL.fit(self, X, y)
        with BaseMatlabLDL.MatlabLDLExec():
            _set_arr('features', X)
            _set_arr('labels', y)
            with io.StringIO() as target:
                eng.AA_BP_fit(nargout=0, stdout=target)
            self._save_net()

    def predict(self, X):
        with BaseMatlabLDL.MatlabLDLExec():
            _set_arr('features', self._X)
            _set_arr('labels', self._D)
            _set_arr('testFeatures', X)
            self._load_net()
            with io.StringIO() as target:
                eng.AA_BP_predict(nargout=0, stdout=target)
            return np.array(eng.workspace['preDistribution'])


class _PT(BaseMatlabLDL):

    def __init__(self, random_state=None):
        super().__init__("model", "model", random_state)


class PT_Bayes(_PT):
    pass


class PT_SVM(_PT):
    pass


__all__ = ["SA_BFGS", "SA_IIS", "AA_KNN", "AA_BP", "PT_Bayes", "PT_SVM"]
