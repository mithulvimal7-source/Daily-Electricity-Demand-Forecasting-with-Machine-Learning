"""Clean DATASET.csv for daily demand forecasting.

Place DATASET.csv beside this script, or pass --input PATH.
Run: python 01_clean_data.py
"""
from pathlib import Path
import argparse
import pandas as pd

TARGET = "TARGET_Next_Day_Demand_(MU)"


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(here / "DATASET.csv"), help="Input CSV path")
    parser.add_argument("--output", default=str(here / "cleaned_DATASET.csv"), help="Output cleaned CSV path")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, encoding="cp1252")
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns or TARGET not in df.columns:
        raise ValueError(f"CSV must contain Date and {TARGET!r} columns.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Convert feature columns to numbers; unparseable entries become missing.
    for col in df.columns:
        if col not in ("Date",):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Date", TARGET]).drop_duplicates().sort_values("Date")
    # One record per date is expected. If conflicting duplicate dates exist, keep the last row.
    df = df.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)

    # Add date-derived numeric features and remove the original timestamp column.
    df["Year_from_Date"] = df["Date"].dt.year
    df["Month_from_Date"] = df["Date"].dt.month
    df["Day_from_Date"] = df["Date"].dt.day
    df["Weekday_from_Date"] = df["Date"].dt.dayofweek
    df["DayOfYear_from_Date"] = df["Date"].dt.dayofyear
    df["Weekend_from_Date"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    df = df.drop(columns="Date")

    # Missing feature values are imputed using the training data's median in each model script.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Input rows: {before:,}")
    print(f"Rows after removing invalid target/date and duplicate dates: {len(df):,}")
    print(f"Missing feature values left for model-time imputation: {int(df.drop(columns=TARGET).isna().sum().sum()):,}")
    print(f"Cleaned data saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
