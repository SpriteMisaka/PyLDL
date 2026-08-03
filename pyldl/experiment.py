from pyldl.utils import load_dataset
from pyldl.metrics import DEFAULT_METRICS


def _resolve_fit_args(algorithm, fit_args):
    kwargs = {}
    for base in reversed(algorithm.__mro__):
        if base.__name__ in fit_args:
            kwargs.update(fit_args[base.__name__])
    return kwargs


def run(
    algorithms, datasets, metrics=None,
    n_folds=10, n_repeats=10, preprocessors=None,
    init_args=None, fit_args=None, random_state=0
):
    import pandas as pd
    from tqdm import tqdm
    from sklearn.model_selection import KFold
    from sklearn.base import TransformerMixin
    from pyldl.algorithms import SSG_LDL
    if metrics is None:
        metrics = DEFAULT_METRICS
    if preprocessors is None:
        preprocessors = [None]
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}

    for preprocessor in preprocessors:
        for dataset in datasets:
            X, D = load_dataset(dataset)

            if preprocessor is not None:
                if isinstance(preprocessor, TransformerMixin):
                    pre_str = f"_{preprocessor.__class__.__name__}"
                elif isinstance(preprocessor, SSG_LDL):
                    pre_str = "_SSG_LDL"
                elif isinstance(preprocessor, list):
                    pre_str = "_Pipeline"
            else:
                pre_str = ""

            for algorithm in algorithms:
                alg_fit_args = _resolve_fit_args(algorithm, fit_args)
                for alg_init_args in init_args.get(algorithm.__name__, [{}]):
                    df = pd.DataFrame(columns=["repeat", "fold"] + metrics)

                    if len(alg_init_args) > 0:
                        init_str = "_".join([f"{k}={v}" for k, v in alg_init_args.items()])
                        init_str = f"_{init_str}"
                    else:
                        init_str = ""

                    setup = f"{algorithm.__name__}{pre_str}{init_str}"
                    tqdm.write(f"Running {setup} on {dataset}")

                    outer_pbar = tqdm(total=n_repeats*n_folds, position=0)
                    inner_pbar = tqdm(total=n_folds, leave=False, position=1, bar_format="{desc}")
                    for i in range(1, n_repeats+1):
                        j = 0
                        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state+i)
                        for train_index, test_index in kfold.split(X):
                            j += 1

                            def _preprocessing(p, X, D):
                                if isinstance(p, TransformerMixin):
                                    X = p.fit_transform(X)
                                elif isinstance(p, SSG_LDL):
                                    X, D = p.fit_transform(X, D)
                                elif isinstance(p, list):
                                    for i in p:
                                        X, D = _preprocessing(i, X, D)
                                return X, D
                            X_train, D_train = _preprocessing(preprocessor, X[train_index], D[train_index])

                            model = algorithm(**alg_init_args)
                            model.fit(X_train, D_train, **alg_fit_args)

                            def _preprocessing_test(p, X):
                                if isinstance(p, TransformerMixin):
                                    X = p.transform(X)
                                elif isinstance(p, SSG_LDL):
                                    pass
                                elif isinstance(p, list):
                                    for i in p:
                                        X = _preprocessing_test(i, X)
                                return X
                            X_test = _preprocessing_test(preprocessor, X[test_index])

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
