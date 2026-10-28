from datetime import date, timedelta
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

TARGET = "TARGET_Next_Day_Demand_(MU)"
MODEL_NAME = "Ridge Regression"
MODEL_FILE = "ridge_demand_model.joblib"


def date_features(day):
    weekday = day.weekday()
    day_of_year = day.timetuple().tm_yday
    return {
        "Year": day.year, "Month": day.month, "Day": day.day,
        "Day_of_Week": weekday, "Is_Weekend": int(weekday >= 5),
        "Day_of_Year": day_of_year, "Year_from_Date": day.year,
        "Month_from_Date": day.month, "Day_from_Date": day.day,
        "Weekday_from_Date": weekday, "DayOfYear_from_Date": day_of_year,
        "Weekend_from_Date": int(weekday >= 5),
    }


def estimate_feature(name, profile, month_profile, latest, df):
    series = pd.to_numeric(profile[name], errors="coerce").dropna() if name in profile else pd.Series(dtype=float)
    if series.empty and name in month_profile:
        series = pd.to_numeric(month_profile[name], errors="coerce").dropna()
    if not series.empty:
        return float(series.mean())
    value = pd.to_numeric(pd.Series([latest.get(name)]), errors="coerce").iloc[0]
    if pd.isna(value):
        value = pd.to_numeric(df[name], errors="coerce").median()
    return float(value)


def show_terminal_prediction(model, feature_columns, df, output_dir):
    latest = df.iloc[-1]
    latest_date = date(int(latest["Year_from_Date"]), int(latest["Month_from_Date"]), int(latest["Day_from_Date"]))
    default_forecast_date = latest_date + timedelta(days=1)

    raw_date = input(f"\nForecast date (YYYY-MM-DD) [{default_forecast_date}]: ").strip()
    if not raw_date:
        forecast_date = default_forecast_date
    else:
        try:
            forecast_date = date.fromisoformat(raw_date)
        except ValueError:
            print("Invalid date. Skipping the interactive prediction.")
            return

    input_date = forecast_date - timedelta(days=1)
    raw_demand = input("Previous day's demand (MU), optional; press Enter to estimate: ").strip()
    if raw_demand:
        try:
            previous_demand = float(raw_demand)
            if previous_demand < 0:
                raise ValueError
            demand_was_estimated = False
        except ValueError:
            print("Enter a non-negative number. Skipping the interactive prediction.")
            return
    else:
        previous_demand = None
        demand_was_estimated = True

    while True:
        raw_weekend = input("Is the required forecast date a weekend? (y/n): ").strip().lower()
        if raw_weekend in ("y", "yes", "n", "no"):
            forecast_weekend = int(raw_weekend in ("y", "yes"))
            break
        print("Please enter y or n.")

    months = pd.to_numeric(df["Month_from_Date"], errors="coerce")
    weekdays = pd.to_numeric(df["Weekday_from_Date"], errors="coerce")
    profile = df[(months == input_date.month) & (weekdays == input_date.weekday())]
    month_profile = df[months == input_date.month]
    date_values = date_features(input_date)
    values = {}
    for name in feature_columns:
        if name in ("Is_Weekend", "Weekend_from_Date"):
            values[name] = forecast_weekend
        elif name in date_values:
            values[name] = date_values[name]
        elif name == "Current_Demand_(MU)" and previous_demand is not None:
            values[name] = previous_demand
        else:
            values[name] = estimate_feature(name, profile, month_profile, latest, df)
            if name == "Current_Demand_(MU)":
                previous_demand = values[name]

    input_row = pd.DataFrame([[values[name] for name in feature_columns]], columns=feature_columns)
    predicted = float(model.predict(input_row)[0])
    print(f"\nPredicted demand for {forecast_date}: {predicted:.2f} MU")
    if demand_was_estimated:
        print(f"Previous-day demand was estimated from historical averages: {previous_demand:.2f} MU")
    result_file = output_dir / "manual_prediction.csv"
    pd.DataFrame([{
        "Forecast_Date": forecast_date.isoformat(),
        "Previous_Day_Demand_MU": previous_demand,
        "Previous_Day_Demand_Estimated": demand_was_estimated,
        "Forecast_Day_Is_Weekend": forecast_weekend,
        "Predicted_Demand_MU": predicted,
    }]).to_csv(result_file, index=False)
    print(f"Prediction saved to: {result_file.resolve()}")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(here / "cleaned_DATASET.csv"))
    parser.add_argument("--output-dir", default=str(here / "ridge_results"))
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
                      ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
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
    print(f"Saved metrics, test predictions, and trained model to: {out.resolve()}")
    show_terminal_prediction(model, list(X.columns), df, out)


if __name__ == "__main__":
    main()
