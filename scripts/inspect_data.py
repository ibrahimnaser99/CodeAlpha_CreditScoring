"""Download and inspect the UCI Credit Card Default dataset."""

from ucimlrepo import fetch_ucirepo


def main() -> None:
    data = fetch_ucirepo(id=350)
    features = data.data.features
    targets = data.data.targets

    print("Dataset variables:")
    print(data.variables.to_string(index=False))
    print(f"\nFeature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print("\nTarget distribution (proportion):")
    print(targets.value_counts(normalize=True))
    print(f"\nMissing feature values: {features.isna().sum().sum()}")
    print(f"Duplicated feature rows: {features.duplicated().sum()}")

    print("\nColumns with 12 or fewer distinct values:")
    for column in features.columns:
        if features[column].nunique() <= 12:
            counts = features[column].value_counts().sort_index().to_dict()
            print(f"{column}: {counts}")


if __name__ == "__main__":
    main()
