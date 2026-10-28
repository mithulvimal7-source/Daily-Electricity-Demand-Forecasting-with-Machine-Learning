from datetime import date, timedelta
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
TARGET = "TARGET_Next_Day_Demand_(MU)"
MODEL_PATH = BASE_DIR / "ridge_results" / "ridge_demand_model.joblib"
DATA_PATH = BASE_DIR / "cleaned_DATASET.csv"
app = Flask(__name__)


def get_date_features(day):
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


def estimate_feature(name, profile, month_profile, latest, history):
    series = pd.to_numeric(profile[name], errors="coerce").dropna() if name in profile else pd.Series(dtype=float)
    if series.empty and name in month_profile:
        series = pd.to_numeric(month_profile[name], errors="coerce").dropna()
    if not series.empty:
        return float(series.mean())
    value = pd.to_numeric(pd.Series([latest.get(name)]), errors="coerce").iloc[0]
    if pd.isna(value):
        value = pd.to_numeric(history[name], errors="coerce").median()
    return float(value)


def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved Ridge model not found: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")
    saved = joblib.load(MODEL_PATH)
    history = pd.read_csv(DATA_PATH)
    if history.empty:
        raise ValueError("Cleaned dataset has no rows.")
    return saved["model"], saved["feature_columns"], history


MODEL, FEATURE_COLUMNS, HISTORY = load_resources()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        forecast_date = date.fromisoformat(str(payload.get("forecast_date", "")))
        raw_demand = str(payload.get("previous_day_demand", "")).strip()
        previous_day_demand = float(raw_demand) if raw_demand else None
        weekend_value = payload.get("is_weekend")
        if weekend_value not in ("yes", "no"):
            raise ValueError("Choose whether the forecast date is a weekend.")
        is_weekend = int(weekend_value == "yes")
        if previous_day_demand is not None and previous_day_demand < 0:
            raise ValueError("Previous-day demand must be zero or greater.")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Please enter valid values."}), 400

    input_date = forecast_date - timedelta(days=1)
    months = pd.to_numeric(HISTORY["Month_from_Date"], errors="coerce")
    weekdays = pd.to_numeric(HISTORY["Weekday_from_Date"], errors="coerce")
    profile = HISTORY[(months == input_date.month) & (weekdays == input_date.weekday())]
    month_profile = HISTORY[months == input_date.month]
    latest = HISTORY.iloc[-1]
    date_values = get_date_features(input_date)
    values = {}

    for name in FEATURE_COLUMNS:
        if name in ("Is_Weekend", "Weekend_from_Date"):
            values[name] = is_weekend
        elif name in date_values:
            values[name] = date_values[name]
        elif name == "Current_Demand_(MU)" and previous_day_demand is not None:
            values[name] = previous_day_demand
        else:
            values[name] = estimate_feature(name, profile, month_profile, latest, HISTORY)
            if name == "Current_Demand_(MU)":
                previous_day_demand = values[name]

    row = pd.DataFrame([[values[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = float(MODEL.predict(row)[0])
    return jsonify({
        "forecast_date": forecast_date.isoformat(),
        "predicted_demand_mu": round(prediction, 2),
        "previous_day_demand_used_mu": round(previous_day_demand, 2),
        "previous_day_demand_was_estimated": not bool(raw_demand),
        "estimated_inputs": "Historical averages for the same month and weekday",
        "weekend_matches_calendar": bool(is_weekend == int(forecast_date.weekday() >= 5)),
    })


if __name__ == "__main__":
    print("Demand forecast app is running at http://127.0.0.1:5000")
    print("Keep this terminal open while using the page. Stop with Ctrl+C.")
    app.run(host="127.0.0.1", port=5000, debug=False)
