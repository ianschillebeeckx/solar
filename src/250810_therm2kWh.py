from pathlib import Path
import pandas as pd

THERM_TO_DELIVERED_KWH = 29.3 * (0.4 * 0.60 / 0.85 + 0.6 * 0.90)  # ≈ 24.09

def load_usage(path: Path) -> pd.DataFrame:
    """Read PG&E usage CSV, parse DATE+START TIME into a single timestamp."""
    df = pd.read_csv(path)
    # Combine date and start-time columns if present
    if "START TIME" in df.columns:
        # Create full datetime from DATE and START TIME
        df["DATE"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["START TIME"].astype(str)
        )
        # Drop time columns if desired
        df = df.drop(columns=["START TIME", "END TIME"] if {"END TIME"}.issubset(df.columns) else ["START TIME"])
    else:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def aggregate_hourly_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-min electricity readings into hourly totals."""
    elec = df[df["TYPE"].str.contains("Electric", case=False)].copy()
    # Floor timestamps to the hour
    elec["HOUR"] = elec["DATE"].dt.floor("H")
    # Sum usage per hour
    hourly = (
        elec.groupby("HOUR", as_index=False)["USAGE (kWh)"].sum()
        .rename(columns={"HOUR": "DATE"})
    )
    # Reattach TYPE column
    hourly["TYPE"] = "Electric"
    return hourly


def daily_gas(df: pd.DataFrame) -> pd.DataFrame:
    """Return distributed daily gas usage in therms."""
    # Normalize full timestamps to date only
    gas_inc = (
        df[df["TYPE"].str.match(r".+gas.*", case=False)]
        .groupby(df["DATE"].dt.normalize(), as_index=True)["USAGE (kWh)"].sum()
    )
    gas_inc = gas_inc.rename("increment_therms").reset_index().rename(columns={"DATE": "DAY"})

    full_days = pd.date_range(gas_inc["DAY"].min(), gas_inc["DAY"].max(), freq="D")
    gas = (
        pd.DataFrame({"DAY": full_days})
        .merge(gas_inc, on="DAY", how="left")
        .fillna({"increment_therms": 0.0})
    )

    dist_rows = []
    span = []
    for _, row in gas.iterrows():
        span.append(row["DAY"])
        inc = row["increment_therms"]
        if inc > 0:
            share = inc / len(span)
            for d in span:
                dist_rows.append({"DAY": d, "gas_therms": share})
            span = []
    for d in span:
        dist_rows.append({"DAY": d, "gas_therms": 0.0})

    gas_daily = pd.DataFrame(dist_rows).sort_values("DAY").reset_index(drop=True)
    return gas_daily


def add_gas_equiv_kwh(hourly: pd.DataFrame, gas_daily: pd.DataFrame,
                      start: pd.Timestamp) -> pd.DataFrame:
    """Return hourly DataFrame with original + gas-adjusted kWh."""
    gas_daily = gas_daily.assign(
        gas_kwh_hourly=lambda g: g["gas_therms"] * THERM_TO_DELIVERED_KWH / 24
    )

    df = hourly.copy()
    # DAY column for merge
    df["DAY"] = df["DATE"].dt.normalize()

    merged = df.merge(
        gas_daily[["DAY", "gas_kwh_hourly"]], on="DAY", how="left"
    )
    # Preserve original usage
    merged["original_kwh"] = merged["USAGE (kWh)"]
    merged["adjusted_kwh"] = merged["original_kwh"]

    # Apply gas equivalent from start date
    mask = merged["DATE"] >= start
    merged.loc[mask, "adjusted_kwh"] += merged.loc[mask, "gas_kwh_hourly"].fillna(0)

    return merged.drop(columns=["DAY", "gas_kwh_hourly"])


if __name__ == "__main__":

    RAW_CSV = Path("../input/pge_usage_2024-08-01_to_2025-07-31.csv")
    OUT_CSV = Path("../output/2025_adjusted_kWh.csv")
    START_DATE = pd.Timestamp("2025-01-01")

    raw = load_usage(RAW_CSV)
    # Aggregate electric 15-min readings to hourly
    hourly_elec = aggregate_hourly_usage(raw)
    # Compute daily gas distribution
    gas_daily = daily_gas(raw)
    # Apply gas-equivalent to hourly data
    adjusted = add_gas_equiv_kwh(hourly_elec, gas_daily, START_DATE)
    # Filter to records on/after START_DATE
    filtered = adjusted[adjusted["DATE"] >= START_DATE]
    filtered.to_csv(OUT_CSV, index=False)
    print(f"✅  Saved adjusted file for period {START_DATE.date()} onwards → {OUT_CSV.resolve()}")
