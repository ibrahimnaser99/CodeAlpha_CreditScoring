import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_data import BASE_DIR, TARGET, clean_data, load_data
from evaluation import best_f1_threshold, ranking_metrics, threshold_metrics
from modeling import build_models
from preprocessing import RANDOM_STATE, add_features, get_feature_columns, split_data, split_xy

REPORTS_DIR = BASE_DIR / "reports"


def main() -> None:
    df, _ = clean_data(load_data())
    df = add_features(df)
    train, val, _test = split_data(df)

    features = get_feature_columns(df)
    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(val, features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    rows = []
    validation_scores = {}
    for name, model in build_models(features).items():
        print(f"Training {name}...")
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        validation_scores[name] = proba

        ranking = ranking_metrics(y_val, proba)
        at_half = threshold_metrics(y_val, proba, 0.5)
        tuned_threshold, tuned_f1 = best_f1_threshold(y_val, proba)
        tuned = threshold_metrics(y_val, proba, tuned_threshold)
        rows.append(
            {
                "model": name,
                "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": cv_scores["test_roc_auc"].std(),
                "cv_pr_auc": cv_scores["test_pr_auc"].mean(),
                "cv_pr_auc_std": cv_scores["test_pr_auc"].std(),
                "val_roc_auc": ranking["roc_auc"],
                "val_pr_auc": ranking["pr_auc"],
                "val_precision@0.5": at_half["precision"],
                "val_recall@0.5": at_half["recall"],
                "val_f1@0.5": at_half["f1"],
                "tuned_threshold": tuned_threshold,
                "val_precision@tuned": tuned["precision"],
                "val_recall@tuned": tuned["recall"],
                "val_f1@tuned": tuned_f1,
                "tn@0.5": at_half["tn"],
                "fp@0.5": at_half["fp"],
                "fn@0.5": at_half["fn"],
                "tp@0.5": at_half["tp"],
            }
        )

    results = pd.DataFrame(rows).set_index("model")
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "baseline_results.csv")

    print("\n=== Cross-validation on train (5 folds, mean +- std) ===")
    for name, row in results.iterrows():
        print(
            f"{name:<20} ROC-AUC {row['cv_roc_auc']:.3f} +- {row['cv_roc_auc_std']:.3f}   "
            f"PR-AUC {row['cv_pr_auc']:.3f} +- {row['cv_pr_auc_std']:.3f}"
        )

    print("\n=== Validation set ===")
    columns = ["val_roc_auc", "val_pr_auc", "val_precision@0.5", "val_recall@0.5", "val_f1@0.5"]
    print(results[columns].round(3).to_string())

    print("\n=== Confusion matrix at threshold 0.5 (validation) ===")
    print(results[["tn@0.5", "fp@0.5", "fn@0.5", "tp@0.5"]].to_string())

    print("\n=== With the threshold tuned for best F1 on validation ===")
    print("(optimistic: the threshold was chosen on the same data)")
    tuned_columns = [
        "tuned_threshold",
        "val_precision@tuned",
        "val_recall@tuned",
        "val_f1@tuned",
    ]
    print(results[tuned_columns].round(3).to_string())

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in validation_scores.items():
        RocCurveDisplay.from_predictions(y_val, proba, name=name, ax=roc_axis)
        PrecisionRecallDisplay.from_predictions(y_val, proba, name=name, ax=pr_axis)
    roc_axis.set_title("ROC curve (validation)")
    pr_axis.set_title("Precision-recall curve (validation)")
    pr_axis.axhline(y_val.mean(), color="grey", linestyle=":", label="no skill")
    pr_axis.legend()
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "baselines_validation_curves.png", dpi=120)
    plt.close()
    print("\nsaved", figures_dir / "baselines_validation_curves.png")
    print("saved", REPORTS_DIR / "baseline_results.csv")


if __name__ == "__main__":
    main()
