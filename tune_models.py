import argparse

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from credit_data import BASE_DIR, clean_data, load_data
from evaluation import ranking_metrics
from preprocessing import RANDOM_STATE, add_features, get_feature_columns, split_data, split_xy
from tuning import SEARCH_N_JOBS, SEARCH_SPACES, build_candidates, save_best_params

REPORTS_DIR = BASE_DIR / "reports"
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def main(n_iter: int) -> None:
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    df, _ = clean_data(load_data())
    df = add_features(df)
    train, val, _test = split_data(df)

    features = get_feature_columns(df)
    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(val, features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    best_params = {}
    for name, pipeline in build_candidates(features).items():
        print(f"Tuning {name} ({n_iter} random combinations x 5 folds)...")
        search = RandomizedSearchCV(
            pipeline,
            SEARCH_SPACES[name],
            n_iter=n_iter,
            scoring=SCORING,
            refit="pr_auc",
            cv=cv,
            n_jobs=SEARCH_N_JOBS[name],
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)

        best = search.best_index_
        results = search.cv_results_
        proba = search.best_estimator_.predict_proba(X_val)[:, 1]
        validation = ranking_metrics(y_val, proba)

        best_params[name] = search.best_params_
        rows.append(
            {
                "model": name,
                "cv_pr_auc": results["mean_test_pr_auc"][best],
                "cv_pr_auc_std": results["std_test_pr_auc"][best],
                "cv_roc_auc": results["mean_test_roc_auc"][best],
                "cv_roc_auc_std": results["std_test_roc_auc"][best],
                "val_pr_auc": validation["pr_auc"],
                "val_roc_auc": validation["roc_auc"],
                "best_params": str(
                    {key.replace("model__", ""): value for key, value in search.best_params_.items()}
                ),
            }
        )

    table = pd.DataFrame(rows).set_index("model")
    REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORTS_DIR / "tuning_results.csv")
    save_best_params(REPORTS_DIR / "best_params.json", best_params)

    print("\n=== Tuned models (CV on train, 5 folds, mean +- std) ===")
    for name, row in table.iterrows():
        print(
            f"{name:<24} PR-AUC {row['cv_pr_auc']:.3f} +- {row['cv_pr_auc_std']:.3f}   "
            f"ROC-AUC {row['cv_roc_auc']:.3f} +- {row['cv_roc_auc_std']:.3f}"
        )

    print("\n=== Validation set ===")
    print(table[["val_pr_auc", "val_roc_auc"]].round(3).to_string())

    print("\n=== Best parameters ===")
    for name, row in table.iterrows():
        print(f"{name}: {row['best_params']}")

    print("\nsaved", REPORTS_DIR / "tuning_results.csv")
    print("saved", REPORTS_DIR / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the credit scoring models with random search.")
    parser.add_argument("--n-iter", type=int, default=20, help="random combinations tried per model")
    parser.add_argument("--quick", action="store_true", help="only 3 combinations per model, to check that it runs")
    args = parser.parse_args()
    main(3 if args.quick else args.n_iter)
