import os
import pickle
import inspect
import logging

from typing import Optional
from functools import wraps
from pathlib import Path

import numpy as np

from sklearn.base import BaseEstimator

from pyldl.algorithms.utils import proj, normalize, DEFAULT_METRICS, DEFAULT_METRICS_GLD


def _path_suffix(suffix):
    def decorator(func):
        @wraps(func)
        def wrapper(self, path: str, *args, **kwargs):
            file = Path(path).with_suffix(suffix)
            os.makedirs(file.parent, exist_ok=True)
            return func(self, file, *args, **kwargs)
        return wrapper
    return decorator


class _Base:
    """Base class for all models in PyLDL.
    """

    def __init__(self, random_state: Optional[int] = None):
        if random_state is not None:
            np.random.seed(random_state)
        self._n_features = None
        self._n_outputs = None
        self._target = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the model.
        """
        self._X = X.astype(np.float64)
        self._n_samples = self._X.shape[0]
        self._n_features = self._X.shape[1]
        self._n_outputs = Y.shape[1]
        return self

    @_path_suffix(".pkl")
    def dump(self, file: str):
        """Save the model to a file.
        """
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    @_path_suffix(".pkl")
    def load(cls, file: str):
        """Load the model from a file.
        """
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, cls):
                return obj
            new_obj = cls.__new__(cls)
            new_obj.__dict__.update(obj.__dict__)
            return new_obj

    def _score_impl(self, X, targets, pred_funcs, default_metrics,
                    metrics, return_dict):
        from pyldl.metrics import NAME_TO_TYPE

        if metrics is None:
            metrics = default_metrics

        def _first_arg_name(func):
            params = list(inspect.signature(inspect.unwrap(func)).parameters.keys())
            return params[0] if params else None

        def _metric_type(metric):
            if isinstance(metric, str):
                t = NAME_TO_TYPE.get(metric)
                if t in pred_funcs:
                    return t
            else:
                arg_name = _first_arg_name(metric)
                if arg_name in pred_funcs:
                    return arg_name
            raise ValueError(f"Unknown metric: {metric}. "
                             "If you are using a custom metric, please make sure that its first argument is "
                             f"named {'/'.join(pred_funcs)} to indicate the type of the metric.")

        preds_cache = {}
        def _predict(t):
            if t not in preds_cache:
                preds_cache[t] = pred_funcs[t](self, X)
            return preds_cache[t]

        import importlib
        module = importlib.import_module('pyldl.metrics')

        scores = []
        for metric in metrics:
            t = _metric_type(metric)
            fn = getattr(module, metric) if isinstance(metric, str) else metric
            scores.append(fn(targets[t], _predict(t)))

        return dict(zip(metrics, scores)) if return_dict else tuple(scores)

    def _not_been_fit(self):
        logging.warning("The model has not yet been fit. "
                        "Try to call 'fit()' first with some training data.")

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            self._not_been_fit()
        return self._n_features

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            self._not_been_fit()
        return self._n_outputs

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_X' in state:
            del state['_X']
        if '_target' in state:
            del state['_target']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._X = None

    def __str__(self) -> str:
        return self.__class__.__name__


class _BaseLDL(_Base):

    def predict(self) -> np.ndarray:
        raise NotImplementedError("The 'predict()' method is not implemented.")

    def fit(self, X: np.ndarray, D: np.ndarray):
        super().fit(X, D)
        self._D = D.astype(np.float64)
        self._target = self._D
        return self

    def score(self, X: np.ndarray, D: np.ndarray,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        from pyldl.algorithms.utils import binaryzation
        targets = {'D': D, 'L': binaryzation(D)}
        pred_funcs = {
            'D': lambda self, X: self.predict(X),
            'L': lambda self, X: binaryzation(self.predict(X)),
        }
        return self._score_impl(X, targets, pred_funcs, DEFAULT_METRICS,
                                metrics, return_dict)

    def __getstate__(self):
        state = super().__getstate__()
        if '_D' in state:
            del state['_D']
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._D = None
        self._target = None


class _BaseLE(_Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._D = None

    def fit(self, X: np.ndarray, L: np.ndarray):
        super().fit(X, L)
        self._L = L.astype(np.float64)
        return self

    def transform(self, *args, **kwargs) -> np.ndarray:
        return self._D

    def fit_transform(self, X: np.ndarray, L: np.ndarray, **kwargs):
        return self.fit(X, L, **kwargs).transform(X, L)

    def score(self, D: np.ndarray, X: np.ndarray = None, L: np.ndarray = None,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        if metrics is None:
            metrics = DEFAULT_METRICS
        from pyldl.metrics import score
        return score(D, self.transform(X, L), metrics=metrics, return_dict=return_dict)

    def __getstate__(self):
        state = super().__getstate__()
        if '_L' in state:
            del state['_L']
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._L = None


class _BaseGLD(_Base):

    def __init__(self, lower_bounds, upper_bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def predict(self, _) -> np.ndarray:
        raise NotImplementedError("The 'predict()' method is not implemented.")

    @staticmethod
    def G2RAW(G: np.ndarray, lb, ub):
        return (ub - lb) * (G + 1.) / 2. + lb

    @staticmethod
    def RAW2G(Q: np.ndarray, lb, ub):
        return 2. * (Q - lb) / (ub - lb) - 1.

    @staticmethod
    def G2D(G: np.ndarray, lb, ub):
        Q = _BaseGLD.G2RAW(G, lb, ub)
        select = np.all(Q >= 0., axis=1) & (np.sum(Q, axis=1) > 0.)
        D = np.empty_like(Q, dtype=np.float64)
        D[select] = normalize(Q[select])
        D[~select] = proj(Q[~select])
        return D

    @staticmethod
    def G2TL(G: np.ndarray):
        return np.where(G > (1. / 3.), 1., np.where(G <= (-1. / 3.), -1., 0.))

    @staticmethod
    def G2L(G: np.ndarray):
        return np.where(G > 0, 1., 0.)

    @staticmethod
    def G2SL(G: np.ndarray):
        return np.argmax(G, axis=1)

    def predict_D(self, X: np.ndarray):
        return _BaseGLD.G2D(self.predict(X), self.lower_bounds, self.upper_bounds)

    def predict_TL(self, X: np.ndarray):
        return _BaseGLD.G2TL(self.predict(X))

    def predict_L(self, X: np.ndarray):
        return _BaseGLD.G2L(self.predict(X))

    def predict_SL(self, X: np.ndarray):
        return _BaseGLD.G2SL(self.predict(X))

    def fit(self, X: np.ndarray, G: np.ndarray):
        super().fit(X, G)
        self._G = G.astype(np.float64)
        self._target = self._G
        return self

    def score(self, X: np.ndarray, G: np.ndarray,
              metrics: Optional[list[str]] = None, return_dict: bool = False):
        targets = {
            'D': self.G2D(G, self.lower_bounds, self.upper_bounds),
            'L': self.G2L(G),
            'G': G,
        }
        pred_funcs = {
            'D': lambda self, X: self.predict_D(X),
            'L': lambda self, X: self.predict_L(X),
            'G': lambda self, X: self.predict(X),
        }
        return self._score_impl(X, targets, pred_funcs, DEFAULT_METRICS_GLD,
                                metrics, return_dict)


class BaseLDL(_BaseLDL, BaseEstimator):
    """Base class for all LDL models in PyLDL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseLE(_BaseLE, BaseEstimator):
    """Base class for all LE models in PyLDL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseGLD(_BaseGLD, BaseEstimator):
    """Base class for all GLD models in PyLDL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Base(_Base):

    def fit(self, X, Y, **kwargs):
        if isinstance(self, BaseIncomLDL):
            mask = kwargs.pop("mask")
            BaseIncomLDL.fit(self, X, Y, mask, **kwargs)
        elif isinstance(self, BaseLDL):
            BaseLDL.fit(self, X, Y, **kwargs)
        elif isinstance(self, BaseLE):
            BaseLE.fit(self, X, Y, **kwargs)
        elif isinstance(self, BaseGLD):
            BaseGLD.fit(self, X, Y, **kwargs)
        else:
            raise TypeError("The model must be a subclass of BaseLDL, BaseLE or BaseGLD.")


class BaseADMM(Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.EPS_ABS = 1e-4
        self.EPS_REL = 1e-3
        self.EPS_ERR = 1e-3
        self._olds = {}

    def _update_W(self):
        pass

    def _update_Z(self):
        pass

    def _update_V(self):
        self._V = self._V + self._rho * (self._X @ self._W - self._Z)

    def _before_train(self):
        pass

    def _get_default_model(self):
        assert self._n_features is not None and self._n_outputs is not None
        _W = np.ones((self._n_features, self._n_outputs))
        _Z = np.ones((self._n_samples, self._n_outputs))
        _V = np.ones((self._n_samples, self._n_outputs))
        return _W, _Z, _V

    @property
    def constraint(self):
        return [[self._Z, self._X @ self._W]]

    @property
    def params(self):
        return [self._W, self._Z]

    @property
    def Vs(self):
        return [self._V]

    def _primal_residual(self):
        return np.array([np.linalg.norm(c[0] - c[1], 'fro') for c in self.constraint])

    def _dual_residual(self):
        return np.array([np.linalg.norm(self._rho * (c[0] - self._olds[i]), 'fro') for i, c in enumerate(self.constraint)])

    def _primal_eps(self):
        return np.sqrt(self._n_samples) * self.EPS_ABS + self.EPS_REL * np.array([np.maximum(
            np.linalg.norm(c[0], 'fro'), np.linalg.norm(c[1], 'fro')
        ) for c in self.constraint])

    def _dual_eps(self):
        assert self._n_outputs is not None
        return np.sqrt(self._n_outputs) * self.EPS_ABS + self.EPS_REL * np.array([
            np.linalg.norm(v, 'fro') for v in self.Vs
        ])

    def _err(self):
        err = 0.
        for i, c in enumerate(self.params):
            err = np.maximum(err, np.abs(c - self._olds[i]).max())
        return err

    def _restore(self):
        if self._stopping_criterion == 'primal_dual':
            for i, c in enumerate(self.constraint):
                self._olds[i] = c[0]
        elif self._stopping_criterion == 'error':
            for i, c in enumerate(self.params):
                self._olds[i] = c

    def _converged(self):
        if self._stopping_criterion == 'primal_dual':
            return np.all(self._primal_residual() <= self._primal_eps()) and np.all(self._dual_residual() <= self._dual_eps())
        elif self._stopping_criterion == 'error':
            return self._err() <= self.EPS_ERR
        elif self._stopping_criterion is None:
            return False

    def fit(self, X, Y, *, max_iterations=100, rho=1., stopping_criterion='primal_dual', **kwargs):
        super().fit(X, Y, **kwargs)
        self._rho = rho
        self._max_iterations = max_iterations
        self._stopping_criterion = stopping_criterion

        self._before_train()

        self._W, self._Z, self._V = self._get_default_model()

        for i in range(self._max_iterations):
            self._current_iteration = i + 1
            self._restore()
            self._update_W()
            self._update_Z()
            self._update_V()
            if self._converged():
                logging.info(f"Converged in {self._current_iteration} iterations.")
                break

        return self

    def predict(self, X):
        return proj(X @ self._W)


class BaseIncomLDL(BaseLDL):
    """Base class for all IncomLDL models in PyLDL.
    """

    @staticmethod
    def repair(D, mask):
        b = np.sum(mask, axis=1)
        c = 1 - np.sum(D, axis=1)
        b0 = b.copy()
        b[b == 0] = 1
        d = c / b
        A = d[:, np.newaxis] * mask
        B = (b0 == 1)[:, np.newaxis] * A
        return D + B, (B == 0) & mask

    def fit(self, X, D, mask):
        super().fit(X, D)
        self._D, mask = self.repair(self._D, mask)
        self._mask = np.where(mask, 0., 1.)


class BaseLDLClassifier(BaseLDL):
    """Base class for all LDL4C models in PyLDL.
    """

    def predict_proba(self):
        raise NotImplementedError("The 'predict_proba()' method is not implemented.")

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, D, metrics=None, return_dict=False):
        if metrics is None:
            metrics = ["zero_one_loss", "error_probability"]
        from pyldl.metrics import score
        return score(D, self.predict_proba(X), metrics=metrics, return_dict=return_dict)


class BaseEnsemble(BaseLDL):

    def __init__(self, estimator: BaseLDL, n_estimators: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = estimator
        self._n_estimators = n_estimators
        self._estimators = None

    def __len__(self):
        return len(self._estimators)

    def __getitem__(self, index):
        return self._estimators[index]

    def __iter__(self):
        return iter(self._estimators)

    @property
    def estimator(self):
        return self._estimator

    @property
    def n_estimators(self):
        return self._n_estimators
