import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from credit_data import clean_data, load_data
from evaluation import (
    brier,
    calibration_table,
    cost_optimal_threshold,
    cost_per_client,
    ranking_metrics,
    threshold_metrics,
)
from final_model import (
    BEST_PARAMS_PATH,
    COST_FP,
    COST_RATIOS,
    CONFIG_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    build_final_estimator,
    load_json,
    save_json,
    save_model,
)
from preprocessing import RANDOM_STATE, add_features, get_feature_columns, split_data, split_xy


def main(model_name: str, cost_ratio: float, calibrate: bool) -> None:
    if cost_ratio <= 0:
        raise ValueError("cost_ratio must be positive.")

    df, _ = clean_data(load_data())
    df = add_features(df)
    train, val, _test = split_data(df)

    features = get_feature_columns(df)
    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(val, features)
    params = load_json(BEST_PARAMS_PATH)

    print(f"Final model: {model_name}")
    models = {
        "raw": build_final_estimator(features, model_name, params, calibrate=False).fit(
            X_train, y_train
        )
    }
    if calibrate:
        print("Calibrating the probabilities (5 extra fits on the training set)...")
        models["calibrated"] = build_final_estimator(
            features, model_name, params, calibrate=True
        ).fit(X_train, y_train)

    probabilities = {
        name: model.predict_proba(X_val)[:, 1] for name, model in models.items()
    }

    print("\n=== Validation: raw scores vs calibrated probabilities ===")
    for name, proba in probabilities.items():
        ranking = ranking_metrics(y_val, proba)
        print(
            f"{name:<11} ROC-AUC {ranking['roc_auc']:.3f}  PR-AUC {ranking['pr_auc']:.3f}  "
            f"Brier {brier(y_val, proba):.4f}  mean predicted {proba.mean():.3f}  "
            f"actual default rate {y_val.mean():.3f}"
        )

    print("\n=== Calibration tables (validation) ===")
    for name, proba in probabilities.items():
        print(f"-- {name}")
        print(calibration_table(y_val, proba).round(3).to_string())

    chosen_name = "calibrated" if calibrate else "raw"
    chosen = models[chosen_name]
    proba = probabilities[chosen_name]

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle=":", color="grey", label="perfect calibration")
    for name, scores in probabilities.items():
        table = calibration_table(y_val, scores)
        plt.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=name)
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed default rate")
    plt.title("Calibration (validation)")
    plt.legend()
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_validation.png", dpi=120)
    plt.close()

    prevalence = float(y_val.mean())
    rows = []
    for ratio in COST_RATIOS:
        threshold, cost = cost_optimal_threshold(
            y_val, proba, cost_fn=ratio, cost_fp=COST_FP
        )
        counts = threshold_metrics(y_val, proba, threshold)
        rows.append(
            {
                "cost_ratio_fn_to_fp": ratio,
                "threshold": threshold,
                "precision": counts["precision"],
                "recall": counts["recall"],
                "false_negatives": counts["fn"],
                "false_positives": counts["fp"],
                "cost_per_client": cost,
                "cost_approve_everyone": ratio * prevalence,
                "cost_reject_everyone": COST_FP * (1 - prevalence),
            }
        )
    sensitivity = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    sensitivity.to_csv(REPORTS_DIR / "cost_sensitivity_validation.csv", index=False)

    print("\n=== Best threshold for different costs (validation) ===")
    print("cost_ratio = how many times worse a missed defaulter is than a rejected good client")
    print(sensitivity.round(3).to_string(index=False))
    print("\nFor well-calibrated probabilities, the best threshold is close to 1 / (1 + cost_ratio):")
    for ratio in COST_RATIOS:
        print(f"  cost_ratio {ratio}: {1 / (1 + ratio):.3f}")

    threshold, cost = cost_optimal_threshold(
        y_val, proba, cost_fn=cost_ratio, cost_fp=COST_FP
    )
    counts = threshold_metrics(y_val, proba, threshold)
    print(f"\nChosen: cost_ratio={cost_ratio}  ->  threshold {threshold:.2f}")
    print(
        f"validation: precision {counts['precision']:.3f}  recall {counts['recall']:.3f}  "
        f"cost per client {cost:.3f}  (approve everyone: "
        f"{cost_per_client(y_val, proba, 1.01, cost_ratio, COST_FP):.3f})"
    )

    save_model(MODEL_PATH, chosen, features)
    save_json(
        CONFIG_PATH,
        {
            "model": model_name,
            "calibrated": calibrate,
            "cost_ratio_fn_to_fp": cost_ratio,
            "cost_fp": COST_FP,
            "threshold": threshold,
            "random_state": RANDOM_STATE,
            "validation": {
                **ranking_metrics(y_val, proba),
                **counts,
                "cost_per_client": cost,
            },
        },
    )
    print("\nsaved", MODEL_PATH)
    print("saved", CONFIG_PATH)
    print("saved", REPORTS_DIR / "cost_sensitivity_validation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the threshold on the validation set.")
    parser.add_argument(
        "--model", default="Random Forest", help="model to use (decide BEFORE looking at the test set)"
    )
    parser.add_argument(
        "--cost-ratio",
        type=float,
        default=3.0,
        help="cost of a missed defaulter / cost of a rejected good client",
    )
    parser.add_argument(
        "--no-calibration", action="store_true", help="use raw scores instead of calibrated probabilities"
    )
    args = parser.parse_args()
    main(args.model, args.cost_ratio, calibrate=not args.no_calibration)
