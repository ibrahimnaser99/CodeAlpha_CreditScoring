import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from credit_data import clean_data, load_data
from final_model import (
    BEST_PARAMS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    build_final_estimator,
    load_json,
    load_model,
)
from preprocessing import RANDOM_STATE, add_features, get_feature_columns, split_data, split_xy


def main(model_name: str) -> None:
    df, _ = clean_data(load_data())
    df = add_features(df)
    train, val, _test = split_data(df)

    features = get_feature_columns(df)
    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(val, features)
    params = load_json(BEST_PARAMS_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)

    model, model_features = load_model(MODEL_PATH)
    if model_features != features:
        raise ValueError(
            "Saved model features do not match the current dataset features. "
            "Run select_final.py again before explaining the model."
        )
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": features,
                "pr_auc_drop": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("pr_auc_drop", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    print(f"=== Permutation importance: {model_name} (validation, drop in PR-AUC) ===")
    print(importance.head(15).round(4).to_string(index=False))

    top = importance.head(15).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top["feature"], top["pr_auc_drop"], xerr=top["std"], color="#4C78A8")
    plt.xlabel("drop in PR-AUC when the column is shuffled")
    plt.title(f"Permutation importance ({model_name}, validation)")
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_importance.png", dpi=120)
    plt.close()

    logistic = build_final_estimator(
        features, "Logistic Regression", params, calibrate=False
    ).fit(X_train, y_train)
    names = logistic.named_steps["prep"].get_feature_names_out()
    coefficients = (
        pd.DataFrame(
            {"feature": names, "coefficient": logistic.named_steps["model"].coef_[0]}
        )
        .assign(abs_coefficient=lambda table: table["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )
    coefficients.to_csv(REPORTS_DIR / "logistic_coefficients.csv", index=False)
    print("\n=== Logistic Regression coefficients (largest first, scaled features) ===")
    print(coefficients.head(15).round(3).to_string(index=False))

    print("\nsaved", REPORTS_DIR / "permutation_importance.csv")
    print("\nsaved", REPORTS_DIR / "logistic_coefficients.csv")
    print("saved", figures_dir / "permutation_importance.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain the final model (no test data used).")
    parser.add_argument("--model", default="Random Forest")
    args = parser.parse_args()
    main(args.model)
