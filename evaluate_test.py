import argparse

from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from credit_data import clean_data, load_data
from evaluation import bootstrap_ci, cost_per_client, ranking_metrics, threshold_metrics
from final_model import (
    CONFIG_PATH,
    MODEL_PATH,
    TEST_RESULTS_PATH,
    ensure_test_not_used,
    load_json,
    load_model,
    save_json,
)
from preprocessing import add_features, split_data, split_xy


def main(force: bool) -> None:
    ensure_test_not_used(TEST_RESULTS_PATH, force)

    config = load_json(CONFIG_PATH)
    estimator, features = load_model(MODEL_PATH)

    df, _ = clean_data(load_data())
    df = add_features(df)
    _train, _val, test = split_data(df)
    X_test, y_test = split_xy(test, features)

    threshold = config["threshold"]
    cost_fn = config["cost_ratio_fn_to_fp"]
    cost_fp = config["cost_fp"]
    proba = estimator.predict_proba(X_test)[:, 1]

    ranking = ranking_metrics(y_test, proba)
    counts = threshold_metrics(y_test, proba, threshold)
    cost = cost_per_client(y_test, proba, threshold, cost_fn, cost_fp)
    prevalence = float(y_test.mean())
    approve_everyone = cost_fn * prevalence
    reject_everyone = cost_fp * (1 - prevalence)

    intervals = {
        "roc_auc": bootstrap_ci(y_test, proba, roc_auc_score),
        "pr_auc": bootstrap_ci(y_test, proba, average_precision_score),
        "recall": bootstrap_ci(
            y_test,
            proba,
            lambda y, p: recall_score(y, p >= threshold, zero_division=0),
        ),
    }

    print(f"=== TEST SET (used once): {config['model']}, calibrated={config['calibrated']} ===")
    print(f"threshold {threshold:.2f} (chosen on validation with cost ratio {cost_fn:g}:1)\n")
    print(
        f"ROC-AUC {ranking['roc_auc']:.3f}  95% CI "
        f"{intervals['roc_auc'][0]:.3f} - {intervals['roc_auc'][1]:.3f}"
    )
    print(
        f"PR-AUC  {ranking['pr_auc']:.3f}  95% CI "
        f"{intervals['pr_auc'][0]:.3f} - {intervals['pr_auc'][1]:.3f}"
    )
    print(
        f"precision {counts['precision']:.3f}   recall {counts['recall']:.3f}  "
        f"(95% CI {intervals['recall'][0]:.3f} - {intervals['recall'][1]:.3f})   "
        f"F1 {counts['f1']:.3f}"
    )
    print(
        f"confusion matrix: TN {counts['tn']}  FP {counts['fp']}  "
        f"FN {counts['fn']}  TP {counts['tp']}"
    )
    print(
        f"\ncost per client: model {cost:.3f}  |  approve everyone {approve_everyone:.3f}  "
        f"|  reject everyone {reject_everyone:.3f}"
    )

    validation = config["validation"]
    print("\n=== Validation vs test (a big gap would be a warning sign) ===")
    for name in ["roc_auc", "pr_auc", "precision", "recall"]:
        test_value = ranking[name] if name in ranking else counts[name]
        print(f"{name:<10} validation {validation[name]:.3f}   test {test_value:.3f}")
    print(f"{'cost':<10} validation {validation['cost_per_client']:.3f}   test {cost:.3f}")

    save_json(
        TEST_RESULTS_PATH,
        {
            "model": config["model"],
            "calibrated": config["calibrated"],
            "threshold": threshold,
            "cost_ratio_fn_to_fp": cost_fn,
            **ranking,
            **counts,
            "cost_per_client": cost,
            "cost_approve_everyone": approve_everyone,
            "cost_reject_everyone": reject_everyone,
            "ci_95": {name: list(interval) for name, interval in intervals.items()},
        },
    )
    print("\nsaved", TEST_RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the test set ONCE with the final model.")
    parser.add_argument(
        "--force", action="store_true", help="run again even though the test set was already used"
    )
    args = parser.parse_args()
    main(args.force)
