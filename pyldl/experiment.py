import sys

from pyldl.utils import load_dataset
from pyldl.metrics import DEFAULT_METRICS


_DVS_PRESETS = {
    **dict.fromkeys(["SJAFFE", "RAF_ML"], ([1., 0., 1., 0., 0., 0.], [0., 1., 0., 1., 1., 1.])),
    "emotion6": ([0., 0., 1., 0., 0., 1., 0.], [1., 1., 0., 1., 1., 0., 0.]),
    "SBU_3DFE": ([1., 0., 0., 0., 0., 1.], [0., 1., 1., 1., 1., 0.]),
    **dict.fromkeys(["Twitter", "Flickr"], ([0., 1., 0., 1., 0., 1., 0., 0.], [1., 0., 0., 0., 1., 0., 1., 1.])),
    "Music": ([1., 0., 1., 0., 1., 1., 1., .5, .5], [0., 1., 0., 1., 0., 0., 0., .5, .5]),
    "Painting": ([1., 0., 0., 1., 0., 1., 0., 0.], [0., 1., 0., 0., 1., 0., 1., 1.]),
    **dict.fromkeys(["M2B", "fbp5500", "SCUT_FBP", "Movie"], ([0., .25, .5, .75, 1.], [1., .75, .5, .25, 0.])),
}


def _is(target, module, name):
    module = sys.modules.get(f"pyldl.algorithms.{module}")
    if module is None:
        return False
    cls = getattr(module, name)
    return issubclass(target, cls) if isinstance(target, type) else isinstance(target, cls)


def _resolve_fit_args(algorithm, fit_args):
    kwargs = {}
    for base in reversed(algorithm.__mro__):
        for targets, values in fit_args.items():
            if base.__name__ in ((targets,) if isinstance(targets, str) else targets):
                kwargs.update(values)
    return kwargs


def _resolve_extra_args(target, extra_args):
    import inspect

    parameters = inspect.signature(target).parameters
    return {
        name: value for name, value in extra_args.items()
        if name in parameters
        and parameters[name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _resolve_dataset_extra_args(extra_args, dataset, algorithm, postprocessor, metrics):
    import numpy as np

    resolved = dict(extra_args)
    for name in ("pos", "neg"):
        if isinstance(resolved.get(name), dict):
            if dataset in resolved[name]:
                resolved[name] = resolved[name][dataset]
            else:
                del resolved[name]
    if dataset in _DVS_PRESETS and (
        _is(algorithm, "_ldl_dvs", "LDL_DVS") or _is(postprocessor, "_divo", "DivO")
        or any(getattr(metric, "__name__", metric) == "divisiveness_error" for metric in metrics)
    ):
        for name, value in zip(("pos", "neg"), _DVS_PRESETS[dataset]):
            resolved.setdefault(name, np.array(value, dtype=np.float32))
    return resolved


def _resolve_metrics(metrics, extra_args):
    from functools import partial
    import pyldl.metrics as metric_module

    resolved = []
    for metric in metrics:
        function = getattr(metric_module, metric) if isinstance(metric, str) else metric
        kwargs = _resolve_extra_args(function, extra_args)
        resolved.append(partial(function, **kwargs) if kwargs else metric)
    return resolved


def _preprocessor_str(preprocessor):
    from sklearn.base import TransformerMixin
    from pyldl.algorithms import SSG_LDL

    if preprocessor is None:
        return ""
    if isinstance(preprocessor, TransformerMixin):
        return f"_{preprocessor.__class__.__name__}"
    if isinstance(preprocessor, SSG_LDL):
        return "_SSG_LDL"
    if isinstance(preprocessor, list):
        return "_Pipeline"


def _preprocessing(preprocessor, X, D):
    from sklearn.base import TransformerMixin
    from pyldl.algorithms import SSG_LDL

    if isinstance(preprocessor, TransformerMixin):
        X = preprocessor.fit_transform(X)
    elif isinstance(preprocessor, SSG_LDL):
        X, D = preprocessor.fit_transform(X, D)
    elif isinstance(preprocessor, list):
        for processor in preprocessor:
            X, D = _preprocessing(processor, X, D)
    return X, D


def _preprocessing_test(preprocessor, X):
    from sklearn.base import TransformerMixin
    from pyldl.algorithms import SSG_LDL

    if isinstance(preprocessor, TransformerMixin):
        X = preprocessor.transform(X)
    elif isinstance(preprocessor, SSG_LDL):
        pass
    elif isinstance(preprocessor, list):
        for processor in preprocessor:
            X = _preprocessing_test(processor, X)
    return X


def _copy_preprocessor(preprocessor):
    import copy

    if preprocessor is None:
        return None
    if isinstance(preprocessor, list):
        return [_copy_preprocessor(p) for p in preprocessor]
    from sklearn.base import TransformerMixin, clone
    if isinstance(preprocessor, TransformerMixin):
        try:
            return clone(preprocessor)
        except (AttributeError, RuntimeError, TypeError):
            return copy.deepcopy(preprocessor)
    return copy.deepcopy(preprocessor)


def _postprocessing(postprocessor, D):
    if callable(postprocessor):
        D = postprocessor(D)
    elif isinstance(postprocessor, list):
        for processor in postprocessor:
            D = _postprocessing(processor, D)
    return D


def _postprocessor_str(postprocessor):
    if postprocessor is None:
        return ""
    if isinstance(postprocessor, list):
        return "_Pipeline"
    if callable(postprocessor):
        name = getattr(postprocessor, "__name__", postprocessor.__class__.__name__)
        return f"_{name}"
    from pyldl.algorithms._divo import DivO
    if isinstance(postprocessor, DivO):
        return "_DivO"


def _wrap_predict(model, postprocessor):
    if postprocessor is None:
        return model
    predict = model.predict

    def predict_with_postprocessing(X):
        return _postprocessing(postprocessor, predict(X))

    model.predict = predict_with_postprocessing
    return model


def _fit_with_extra_args(model, X, D, fit_args, extra_args):
    original_score = model.score
    instance_score = model.__dict__.get("score")
    has_instance_score = "score" in model.__dict__
    validation_name = "_calculate_validation_scores"
    original_validation = getattr(model, validation_name, None)
    instance_validation = model.__dict__.get(validation_name)
    has_instance_validation = validation_name in model.__dict__

    def score_with_extra_args(X, D, metrics=None, return_dict=False):
        if metrics is None:
            return original_score(X, D, metrics=metrics, return_dict=return_dict)
        resolved = _resolve_metrics(metrics, extra_args)
        values = original_score(X, D, metrics=resolved, return_dict=False)
        return dict(zip(metrics, values)) if return_dict else values
    model.score = score_with_extra_args

    if original_validation is not None:
        def validation_with_extra_args(*args, **kwargs):
            metrics = model._metrics
            resolved = _resolve_metrics(metrics, extra_args)
            model._metrics = resolved
            try:
                scores = original_validation(*args, **kwargs)
                if not scores:
                    return scores
            finally:
                model._metrics = metrics
            return {metric: scores[bound] for metric, bound in zip(metrics, resolved)}
        model._calculate_validation_scores = validation_with_extra_args

    try:
        return model.fit(X, D, **fit_args)
    finally:
        if has_instance_score:
            model.score = instance_score
        else:
            del model.score
        if original_validation is not None:
            if has_instance_validation:
                model._calculate_validation_scores = instance_validation
            else:
                del model._calculate_validation_scores


def _run_fold(
    X, D, train_index, test_index, repeat, fold,
    algorithm, alg_init_args, alg_fit_args, preprocessor, postprocessor,
    post_fit_args, metrics, extra_args, model_path, base_model_path,
    load_models, save_models
):
    from pathlib import Path

    X_train, D_train = _preprocessing(preprocessor, X[train_index], D[train_index])
    X_test = _preprocessing_test(preprocessor, X[test_index])
    loader = algorithm
    if postprocessor is not None and not callable(postprocessor) and not isinstance(postprocessor, list):
        from pyldl.algorithms._divo import DivO
        if isinstance(postprocessor, DivO):
            loader = type(postprocessor)

    model_exists = model_path is not None and any(
        Path(f"{model_path}{suffix}").exists()
        for suffix in (".pkl", ".keras")
    )
    if load_models and model_exists:
        model = loader.load(model_path)
        if loader is not algorithm:
            postprocessor = None
    else:
        base_model_exists = base_model_path is not None and any(
            Path(f"{base_model_path}{suffix}").exists()
            for suffix in (".pkl", ".keras")
        )
        if load_models and base_model_exists:
            model = algorithm.load(base_model_path)
        else:
            model = algorithm(**{
                **_resolve_extra_args(algorithm, extra_args),
                **alg_init_args,
            })
            _fit_with_extra_args(model, X_train, D_train, alg_fit_args, extra_args)
            if save_models and loader is not algorithm:
                model.dump(base_model_path)
        if loader is not algorithm:
            import copy
            postprocessor = copy.deepcopy(postprocessor)
            for name, value in _resolve_extra_args(type(postprocessor), extra_args).items():
                setattr(postprocessor, name, value)
            postprocessor.init_model = model
            model = _fit_with_extra_args(
                postprocessor, X_train, D_train, post_fit_args, extra_args
            )
            postprocessor = None
        if save_models:
            model.dump(model_path)
    model = _wrap_predict(model, postprocessor)
    scores = model.score(X_test, D[test_index], metrics=metrics)
    return repeat, fold, scores


def run(
    algorithms, datasets, metrics=None, *,
    n_folds=10, n_repeats=10, preprocessors=None, postprocessors=None,
    init_args=None, fit_args=None, random_state=0, extra_args=None,
    save_models=False, load_models=False, n_jobs=1
):
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from sklearn.model_selection import KFold
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer or -1.")
    n_jobs = int(n_jobs)
    if metrics is None:
        metrics = DEFAULT_METRICS
    if preprocessors is None:
        preprocessors = [None]
    if postprocessors is None:
        postprocessors = [None]
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}
    if extra_args is None:
        extra_args = {}

    for preprocessor in preprocessors:
        for postprocessor in postprocessors:
            for dataset in datasets:
                X, D = load_dataset(dataset)

                pre_str = _preprocessor_str(preprocessor)
                post_str = _postprocessor_str(postprocessor)

                for algorithm in algorithms:
                    alg_fit_args = _resolve_fit_args(algorithm, fit_args)
                    alg_extra_args = _resolve_dataset_extra_args(extra_args, dataset, algorithm, postprocessor, metrics)
                    score_metrics = _resolve_metrics(metrics, alg_extra_args)
                    for alg_init_args in init_args.get(algorithm.__name__, [{}]):
                        post_fit_args = {}
                        if postprocessor is not None and not callable(postprocessor) and not isinstance(postprocessor, list):
                            from pyldl.algorithms._divo import DivO
                            if isinstance(postprocessor, DivO):
                                post_fit_args = _resolve_fit_args(type(postprocessor), fit_args)
                        if len(alg_init_args) > 0:
                            init_str = "_".join([f"{k}={v}" for k, v in alg_init_args.items()])
                            init_str = f"_{init_str}"
                        else:
                            init_str = ""

                        base_setup = f"{algorithm.__name__}{pre_str}{init_str}"
                        setup = f"{algorithm.__name__}{pre_str}{post_str}{init_str}"
                        tqdm.write(f"Running {setup} on {dataset}")

                        def tasks():
                            for i in range(1, n_repeats + 1):
                                kfold = KFold(
                                    n_splits=n_folds,
                                    shuffle=True,
                                    random_state=random_state + i
                                )
                                for j, (train_index, test_index) in enumerate(kfold.split(X), 1):
                                    task_preprocessor = (
                                        _copy_preprocessor(preprocessor)
                                        if n_jobs != 1 else preprocessor
                                    )
                                    yield delayed(_run_fold)(
                                        X,
                                        D,
                                        train_index,
                                        test_index,
                                        i,
                                        j,
                                        algorithm,
                                        alg_init_args,
                                        alg_fit_args,
                                        task_preprocessor,
                                        postprocessor,
                                        post_fit_args,
                                        score_metrics,
                                        alg_extra_args,
                                        f"models/{algorithm.__name__}/{dataset}/{setup}_repeat={i}_fold={j}"
                                        if save_models or load_models else None,
                                        f"models/{algorithm.__name__}/{dataset}/{base_setup}_repeat={i}_fold={j}"
                                        if save_models or load_models else None,
                                        load_models,
                                        save_models,
                                    )

                        total = n_repeats * n_folds
                        results = []
                        score_sums = [0.] * len(metrics)
                        outer_pbar = tqdm(total=total, position=0)
                        inner_pbar = tqdm(
                            total=n_folds,
                            leave=False,
                            position=1,
                            bar_format="{desc}",
                        )
                        try:
                            with Parallel(
                                n_jobs=n_jobs,
                                backend="loky",
                                return_as="generator",
                            ) as parallel:
                                for repeat, fold, scores in parallel(tasks()):
                                    result = repeat, fold, scores
                                    results.append(result)
                                    score_sums = [
                                        total + score
                                        for total, score in zip(score_sums, scores)
                                    ]
                                    outer_pbar.update(1)
                                    inner_pbar.set_description_str(
                                        f"[repeat {repeat}/{n_repeats}, fold {fold}/{n_folds}] "
                                        + " | ".join(
                                            f"{metric}: {total / len(results):.4f}"
                                            for metric, total in zip(metrics, score_sums)
                                        )
                                        + " "
                                    )
                        finally:
                            outer_pbar.close()
                            inner_pbar.close()

                        results.sort(key=lambda result: (result[0], result[1]))
                        rows = [
                            [repeat, fold] + list(scores)
                            for repeat, fold, scores in results
                        ]
                        df = pd.DataFrame(rows, columns=["repeat", "fold"] + metrics)
                        means = df[metrics].mean()
                        stds = df[metrics].std()
                        df.loc[len(df.index)] = [""] * len(df.columns)
                        df.loc[len(df.index)] = ["", "mean"] + means.tolist()
                        df.loc[len(df.index)] = ["", "std"] + stds.tolist()
                        df.to_csv(f"{setup}_{dataset}.csv", index=False)
                        tqdm.write("(Done!)")
