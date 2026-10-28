from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

TARGET = "TARGET_Next_Day_Demand_(MU)"
MODEL_NAME = "Random Forest"
MODEL_FILE = "random_forest_demand_model.joblib"


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(here / "cleaned_DATASET.csv"))
    parser.add_argument("--output-dir", default=str(here / "random_forest_results"))
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
                      ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                      random_state=42, n_jobs=-1))])
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


if __name__ == "__main__":
    main()
