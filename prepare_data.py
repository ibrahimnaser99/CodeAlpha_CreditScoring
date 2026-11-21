import pandas as pd

from credit_data import TARGET, clean_data, load_data
from preprocessing import ENGINEERED_FEATURES, add_features, get_feature_columns, split_data


def main() -> None:
    df, _ = clean_data(load_data())
    df = add_features(df)
    train, val, test = split_data(df)

    print("=== Split sizes and default rate ===")
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        print(f"{name:<11} rows={len(part):>6}  default_rate={part[TARGET].mean():.4f}")

    features = get_feature_columns(df)
    print(f"\n=== {len(features)} model features ===")
    print(features)

    print("\n=== Engineered features by outcome (train only, mean) ===")
    print(train.groupby(TARGET)[ENGINEERED_FEATURES].mean().round(3).T.to_string())

    print("\n=== Correlation with the target (train only) ===")
    print(train[ENGINEERED_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).round(3).to_string())

    print("\n=== Engineered features summary (train only) ===")
    print(train[ENGINEERED_FEATURES].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
