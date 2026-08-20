from pyldl.utils import load_dataset
from pyldl.metrics import DEFAULT_METRICS


def _resolve_fit_args(algorithm, fit_args):
    kwargs = {}
    for base in reversed(algorithm.__mro__):
        if base.__name__ in fit_args:
            kwargs.update(fit_args[base.__name__])
    return kwargs


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


def _wrap_predict(model, postprocessor):
    if postprocessor is None:
        return model
    predict = model.predict

    def predict_with_postprocessing(X):
        return _postprocessing(postprocessor, predict(X))

    model.predict = predict_with_postprocessing
    return model


def run(
    algorithms, datasets, metrics=None,
    n_folds=10, n_repeats=10, preprocessors=None, postprocessors=None,
    init_args=None, fit_args=None, random_state=0
):
    import pandas as pd
    from tqdm import tqdm
    from sklearn.model_selection import KFold
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

    for preprocessor in preprocessors:
        for postprocessor in postprocessors:
            for dataset in datasets:
                X, D = load_dataset(dataset)

                pre_str = _preprocessor_str(preprocessor)
                post_str = _postprocessor_str(postprocessor)

                for algorithm in algorithms:
                    alg_fit_args = _resolve_fit_args(algorithm, fit_args)
                    for alg_init_args in init_args.get(algorithm.__name__, [{}]):
                        df = pd.DataFrame(columns=["repeat", "fold"] + metrics)

                        if len(alg_init_args) > 0:
                            init_str = "_".join([f"{k}={v}" for k, v in alg_init_args.items()])
                            init_str = f"_{init_str}"
                        else:
                            init_str = ""

                        setup = f"{algorithm.__name__}{pre_str}{post_str}{init_str}"
                        tqdm.write(f"Running {setup} on {dataset}")

                        outer_pbar = tqdm(total=n_repeats*n_folds, position=0)
                        inner_pbar = tqdm(total=n_folds, leave=False, position=1, bar_format="{desc}")
                        for i in range(1, n_repeats+1):
                            j = 0
                            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state+i)
                            for train_index, test_index in kfold.split(X):
                                j += 1

                                X_train, D_train = _preprocessing(
                                    preprocessor,
                                    X[train_index],
                                    D[train_index]
                                )

                                model = algorithm(**alg_init_args)
                                model.fit(X_train, D_train, **alg_fit_args)

                                X_test = _preprocessing_test(
                                    preprocessor,
                                    X[test_index]
                                )

                                model = _wrap_predict(model, postprocessor)
                                scores = model.score(X_test, D[test_index], metrics=metrics)
                                df.loc[len(df.index)] = (i, j) + scores

                                means = df[metrics].mean()
                                stds = df[metrics].std()
                                outer_pbar.update(1)
                                inner_pbar.set_description_str(
                                    f"[repeat {i}/{n_repeats}, fold {j}/{n_folds}] " +
                                    " | ".join(f"{m}: {v:.4f}" for m, v in means.items()) +
                                    " "
                                )

                        df.loc[len(df.index)] = [""] * len(df.columns)
                        df.loc[len(df.index)] = ["", "mean"] + means.tolist()
                        df.loc[len(df.index)] = ["", "std"] + stds.tolist()
                        
                        df.to_csv(f"{setup}_{dataset}.csv", index=False)

                        outer_pbar.close()
                        inner_pbar.close()
                        tqdm.write("(Done!)")
