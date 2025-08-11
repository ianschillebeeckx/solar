from pathlib import Path
import pandas as pd
from datetime import datetime


THERM_TO_DELIVERED_KWH = 29.3 * (0.4 * 0.60 / 0.85 + 0.6 * 0.90)  # ≈ 24.09


def load_usage(path: Path) -> pd.DataFrame:
    """Read PG&E usage CSV and parse DATE."""
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def daily_gas(df: pd.DataFrame) -> pd.DataFrame:
    """Return **distributed** daily gas usage in therms.

    The meter only registers whole‑therm increments. Whenever a non‑zero
    increment appears, it is assumed to represent the cumulative gas used
    since the *previous* increment (or since the start of the file).
    That increment is therefore spread evenly over the run of days that
    precede **and include** the increment day.

    Example
    -------
    Raw increments : 0, 0, 1.03, 0, 1.02, 0, 0, 0, 1.00
    Distributed    : 0.34, 0.34, 0.34, 0.51, 0.51, 0.25, 0.25, 0.25, 0.25
    """
    # Aggregate raw daily increments (may be 0 most days)
    gas_inc = (
        df[df["TYPE"].str.match(r".+gas.*", case=False)]
        .groupby(df["DATE"].dt.normalize(), as_index=True)["USAGE (kWh)"]
        .sum()
    )
    gas_inc = pd.DataFrame(gas_inc).reset_index().rename(columns={"DATE": "DAY", "USAGE (kWh)": "increment_therms"})


    # Ensure every calendar day is present
    full_days = pd.date_range(gas_inc["DAY"].min(), gas_inc["DAY"].max(), freq="D")
    gas = (
        pd.DataFrame({"DAY": full_days})
        .merge(gas_inc, on="DAY", how="left")
        .fillna({"increment_therms": 0.0})
    )

    # Distribute each non‑zero increment backward over the run of days
    # since the last increment (inclusive).
    dist_rows = []
    span = []  # list of days since last increment (running)
    for _, row in gas.iterrows():
        span.append(row["DAY"])
        inc = row["increment_therms"]
        if inc > 0:  # distribute now
            share = inc / len(span)
            for d in span:
                dist_rows.append({"DAY": d, "gas_therms": share})
            span = []  # reset span

    # Any trailing days after the last increment get zero usage
    for d in span:
        dist_rows.append({"DAY": d, "gas_therms": 0.0})

    gas_daily = pd.DataFrame(dist_rows).sort_values("DAY").reset_index(drop=True)
    return gas_daily



def add_gas_equiv_kwh(hourly: pd.DataFrame, gas_daily: pd.DataFrame,
                      start: pd.Timestamp) -> pd.DataFrame:
    """Return hourly DataFrame with both the original and gas‑adjusted kWh.

    * ``USAGE (kWh)`` – raw value from the smart‑meter file (left untouched).
    * ``adjusted_kwh`` – same value plus the gas‑to‑electric equivalent for
      electric rows dated ``start`` or later.
    """
    gas_daily = gas_daily.assign(
        gas_kwh_hourly=lambda g: g["gas_therms"] * THERM_TO_DELIVERED_KWH / 24
    )

    hourly = hourly.copy()
    hourly["DAY"] = hourly["DATE"].dt.normalize()

    merged = hourly.merge(gas_daily[["DAY", "gas_kwh_hourly"]], on="DAY", how="left")

    # Initialise adjusted_kwh as the unmodified usage
    merged["adjusted_kwh"] = merged["USAGE (kWh)"]

    # Only add gas equivalence to electric rows on/after START_DATE
    mask = (
        merged["TYPE"].str.contains("Electric", case=False)
        & (merged["DATE"] >= start)
    )
    merged.loc[mask, "adjusted_kwh"] += merged.loc[mask, "gas_kwh_hourly"]

    return merged.drop(columns=["DAY", "gas_kwh_hourly"])


if __name__ == "__main__":
    RAW_CSV = Path("../input/pge_usage_2024-08-01_to_2025-07-31.csv")
    OUT_CSV = Path("../output/2025_adjusted_kWh.csv")
    START_DATE = pd.Timestamp("2025-01-01")

    raw = load_usage(RAW_CSV)
    gas_daily = daily_gas(raw)
    adjusted = add_gas_equiv_kwh(raw, gas_daily, START_DATE)
    adjusted.to_csv(OUT_CSV, index=False)
    print(f"✅  Saved adjusted file → {OUT_CSV.resolve()}")