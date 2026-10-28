"""Predict next-day demand with only three user inputs.

Run: python 05_manual_prediction.py
The model is the saved Ridge Regression model. Other features are estimated
from historical averages in cleaned_DATASET.csv, grouped by month and weekday.
"""
from datetime import date, timedelta
from pathlib import Path
import joblib
import pandas as pd

TARGET = "TARGET_Next_Day_Demand_(MU)"


def date_features(day):
    """Date features must describe the known-input day before the forecast date."""
    weekday = day.weekday()  # Monday=0 ... Sunday=6
    weekend = int(weekday >= 5)
    day_of_year = day.timetuple().tm_yday
    return {
        "Year": day.year,
        "Month": day.month,
        "Day": day.day,
        "Day_of_Week": weekday,
        "Is_Weekend": weekend,
        "Day_of_Year": day_of_year,
        "Year_from_Date": day.year,
        "Month_from_Date": day.month,
        "Day_from_Date": day.day,
        "Weekday_from_Date": weekday,
        "DayOfYear_from_Date": day_of_year,
        "Weekend_from_Date": weekend,
    }


def main():
    here = Path(__file__).resolve().parent
    model_path = here / "ridge_results" / "ridge_demand_model.joblib"
    data_path = here / "cleaned_DATASET.csv"
    if not model_path.exists():
        raise FileNotFoundError("Saved Ridge model not found. Run 02_train_ridge.py first.")
    if not data_path.exists():
        raise FileNotFoundError("cleaned_DATASET.csv not found. Run 01_clean_data.py first.")

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    history = pd.read_csv(data_path)
    if history.empty:
        raise ValueError("cleaned_DATASET.csv has no rows.")

    # Use the latest known row only to provide sensible fallback values if a month/weekday
    # has no historical match. The rest are estimated from historical month/weekday means.
    latest = history.iloc[-1]
    raw_date = input("Forecast date (YYYY-MM-DD): ").strip()
    try:
        forecast_date = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise ValueError("Enter a valid date in YYYY-MM-DD format, for example 2026-10-24.") from exc

    while True:
        raw_demand = input("Demand on the previous day (MU): ").strip()
        try:
            previous_day_demand = float(raw_demand)
            break
        except ValueError:
            print("Enter the previous day's demand as a number, for example 290.5.")

    input_date = forecast_date - timedelta(days=1)
    actual_weekend = forecast_date.weekday() >= 5
    while True:
        raw_weekend = input("Is the required forecast day a weekend? (y/n): ").strip().lower()
        if raw_weekend in ("y", "yes", "n", "no"):
            weekend = int(raw_weekend in ("y", "yes"))
            break
        print("Please enter y or n.")
    if weekend != int(actual_weekend):
        print(f"Note: {forecast_date} is {'a weekend' if actual_weekend else 'a weekday'} by the calendar; using your answer.")

    # Historical monthly + weekday averages automatically estimate weather and other
    # unentered values. The demand on the previous day is supplied by the user.
    history["_month"] = pd.to_numeric(history["Month_from_Date"], errors="coerce")
    history["_weekday"] = pd.to_numeric(history["Weekday_from_Date"], errors="coerce")
    profile = history[(history["_month"] == input_date.month) &
                      (history["_weekday"] == input_date.weekday())]
    month_profile = history[history["_month"] == input_date.month]
    date_values = date_features(input_date)

    values = {}
    for name in feature_columns:
        if name in ("Is_Weekend", "Weekend_from_Date"):
            values[name] = weekend
        elif name in date_values:
            values[name] = date_values[name]
        elif name == "Current_Demand_(MU)":
            values[name] = previous_day_demand
        else:
            # Average matching calendar rows; fall back to that month's mean, then the
            # latest dataset value if the column has no usable historical values.
            series = pd.to_numeric(profile[name], errors="coerce").dropna() if name in profile else pd.Series(dtype=float)
            if series.empty and name in month_profile:
                series = pd.to_numeric(month_profile[name], errors="coerce").dropna()
            if series.empty:
                value = pd.to_numeric(pd.Series([latest.get(name)]), errors="coerce").iloc[0]
                if pd.isna(value):
                    value = pd.to_numeric(history[name], errors="coerce").median()
            else:
                value = series.mean()
            values[name] = float(value)

    row = pd.DataFrame([[values[name] for name in feature_columns]], columns=feature_columns)
    predicted_demand = float(model.predict(row)[0])
    print(f"\nPredicted demand for {forecast_date}: {predicted_demand:.2f} MU")
    print("Other model inputs were estimated from historical values for the same month and weekday.")

    result_path = here / "manual_prediction.csv"
    pd.DataFrame([{
        "Forecast_Date": forecast_date.isoformat(),
        "Previous_Day_Demand_MU": previous_day_demand,
        "Forecast_Day_Is_Weekend": weekend,
        "Predicted_Demand_MU": predicted_demand,
    }]).to_csv(result_path, index=False)
    print(f"Saved prediction to: {result_path}")


if __name__ == "__main__":
    main()


