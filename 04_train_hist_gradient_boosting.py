from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

TARGET = "TARGET_Next_Day_Demand_(MU)"
MODEL_NAME = "Histogram Gradient Boosting"
MODEL_FILE = "hist_gradient_boosting_demand_model.joblib"


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(here / "cleaned_DATASET.csv"))
    parser.add_argument("--output-dir", default=str(here / "gradient_boosting_results"))
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column {TARGET!r}. Run 01_clean_data.py first.")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    X, y = df.drop(columns=TARGET), df[TARGET]
    split = int(len(df) * 0.8)
    if split < 1 or split >= len(df):
        raise ValueError("Need at least two usable rows.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                                              random_state=42))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"Model": MODEL_NAME, "MAE_MU": mean_absolute_error(y_test, pred),
               "RMSE_MU": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred),
               "Train_rows": len(X_train), "Test_rows": len(X_test)}
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({"Actual_MU": y_test.to_numpy(), "Predicted_MU": pred}).to_csv(out / "test_predictions.csv", index=False)
    joblib.dump({"model": model, "feature_columns": list(X.columns), "target": TARGET}, out / MODEL_FILE)
    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Saved metrics, predictions, and model to: {out.resolve()}")

    # Run this script last. It gathers the three saved model metrics and plots them.
    result_dirs = [here / "ridge_results", here / "random_forest_results", here / "gradient_boosting_results"]
    metric_files = [folder / "metrics.csv" for folder in result_dirs]
    if all(path.exists() for path in metric_files):
        comparison = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
        comparison.to_csv(here / "model_comparison.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        chart_specs = [("MAE_MU", "MAE (MU) — lower is better", "#3274A1"),
                       ("RMSE_MU", "RMSE (MU) — lower is better", "#E1812C"),
                       ("R2", "R² — higher is better", "#3A923A")]
        for ax, (column, title, color) in zip(axes, chart_specs):
            bars = ax.bar(comparison["Model"], comparison[column], color=color)
            ax.set_title(title)
            ax.tick_params(axis="x", labelrotation=25)
            ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.margins(y=0.15)
        fig.suptitle("Next-Day Demand Model Comparison (Chronological Test Set)")
        fig.tight_layout()
        chart_path = here / "model_comparison.png"
        fig.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison table: {(here / 'model_comparison.csv').resolve()}")
        print(f"Comparison graph: {chart_path.resolve()}")
    else:
        print("To create the comparison graph, run the other two model scripts first, then run this script again.")


if __name__ == "__main__":
    main()



