import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from credit_data import BASE_DIR, TARGET, clean_data, load_data

FIGURES_DIR = BASE_DIR / "reports" / "figures"


def default_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return each group's default rate and client count."""
    return df.groupby(column, observed=True)[TARGET].agg(default_rate="mean", clients="size").round(3)


def save_figure(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print("saved", path)


def main() -> None:
    df, report = clean_data(load_data())

    print("=== Cleaning report ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n=== Target ===")
    print(df[TARGET].value_counts().to_string())
    print(df[TARGET].value_counts(normalize=True).round(4).to_string())

    print("\n=== Numeric summary ===")
    print(df.describe().T.round(1).to_string())

    df["age_group"] = pd.cut(df["age"], bins=[20, 30, 40, 50, 60, 100], right=False)
    for column in ["sex", "education", "marriage", "age_group", "pay_1"]:
        print(f"\n=== Default rate by {column} ===")
        print(default_rate_by(df, column).to_string())

    df[TARGET].value_counts().sort_index().plot(kind="bar", color=["#4C78A8", "#E45756"])
    plt.title("Target distribution (0 = paid, 1 = default)")
    plt.xlabel(TARGET)
    plt.ylabel("clients")
    plt.xticks(rotation=0)
    save_figure("target_distribution.png")

    default_rate_by(df, "pay_1")["default_rate"].plot(kind="bar", color="#E45756")
    plt.title("Default rate by most recent repayment status (pay_1)")
    plt.xlabel("pay_1")
    plt.ylabel("default rate")
    save_figure("default_rate_by_pay_1.png")

    for value, label in [(0, "paid"), (1, "default")]:
        plt.hist(df.loc[df[TARGET] == value, "limit_bal"], bins=40, alpha=0.6, density=True, label=label)
    plt.title("Credit limit by outcome")
    plt.xlabel("limit_bal")
    plt.ylabel("density")
    plt.legend()
    save_figure("limit_bal_by_default.png")

    numeric = df.drop(columns=["age_group"])
    correlation = numeric.corr()
    plt.figure(figsize=(11, 9))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation)), correlation.columns)
    plt.title("Correlation matrix")
    save_figure("correlation_matrix.png")


if __name__ == "__main__":
    main()
