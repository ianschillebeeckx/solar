from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === Conversion constants ===
# Based on actual usage data (Dec 12 2025 switch from gas to heat pump water heater):
#   - Historical gas: 0.276 therms/day
#   - Post-switch (cooking only): 0.056 therms/day (20%)
#   - Water heating: 0.220 therms/day (80%)
#
# === What this script models ===
# - Water heating: CONVERTED to electricity (heat pump water heater)
# - Cooking: STAYS AS GAS (not converted)
#
# === Future enhancement ===
# To model cooking with induction, set CONVERT_COOKING = True and add
# cooking_kwh to the output. Use THERM_COOKING_TO_KWH for conversion.
# Cooking hours would typically be 17:00-18:00 (dinner prep).

# End-use fractions (from real data)
WATER_HEAT_FRACTION = 0.80  # 80% of gas was water heating
COOKING_FRACTION = 0.20     # 20% of gas is cooking

# Efficiency assumptions
GAS_WATER_HEATER_EFF = 0.90   # tankless gas water heater efficiency
HEAT_PUMP_COP = 3.9           # heat pump water heater COP
INDUCTION_EFF = 0.87          # induction cooktop efficiency
GAS_COOKING_EFF = 0.38        # gas cooktop efficiency

# Conversion: kWh per therm for each end-use
# Formula: 29.3 kWh/therm * fraction * gas_efficiency / electric_efficiency
THERM_WATER_TO_KWH = 29.3 * WATER_HEAT_FRACTION * GAS_WATER_HEATER_EFF / HEAT_PUMP_COP  # ≈ 5.4 kWh per therm
THERM_COOKING_TO_KWH = 29.3 * COOKING_FRACTION * GAS_COOKING_EFF / INDUCTION_EFF        # ≈ 2.6 kWh per therm (unused for now)

# Schedule: water heating hours (heat pump runs midday to use solar)
WATER_HEAT_HOURS = (11, 12, 13)  # three hours at midday

# === EV charging ===
# NOTE: EV charging logic has moved to sim_batteries.py where it can be
# scheduled dynamically based on surplus solar generation.
# These constants are kept for reference:
# EV_CHARGE_DAYS = (6, 2, 4, 5)  # Sun, Wed, Fri, Sat (dayofweek: Mon=0..Sun=6)
# EV_CHARGE_HOURS = (10, 11, 12, 13, 14)  # 10am-2pm
# EV_WEEKLY_KWH = 46.0


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
    """Return distributed daily gas usage in therms.

    Notes
    -----
    This keeps your original logic that spreads meter-read increments (increment_therms)
    backward over the span since the last positive increment.
    """
    gas_only = df[df["TYPE"].str.match(r".+gas.*", case=False)].copy()
    gas_only["DAY"] = gas_only["DATE"].dt.normalize()

    gas_inc = (
        gas_only.groupby("DAY", as_index=True)["USAGE (kWh)"].sum()
        .rename("increment_therms").reset_index()
    )

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


def add_gas_equiv_kwh(
    hourly: pd.DataFrame,
    gas_daily: pd.DataFrame,
    start: pd.Timestamp,
    water_hours: tuple[int, ...] = WATER_HEAT_HOURS,
) -> pd.DataFrame:
    """Return hourly DataFrame with separate columns for base and water heating kWh.

    Allocation
    ----------
    * Water heating (80% of gas): converted to electricity via heat pump COP,
      split evenly across ``water_hours`` (default 11:00, 12:00, 13:00).
    * Cooking (20% of gas): stays as gas, NOT converted to electricity.

    Output columns
    --------------
    * base_kwh: original electricity usage
    * water_heat_kwh: converted water heating load
    * total_kwh: sum of base + water heating
    """
    # Compute per-day water heating kWh (only water heating is converted)
    g = gas_daily.copy()
    g["water_kwh_day"] = g["gas_therms"] * THERM_WATER_TO_KWH
    g["water_kwh_per_hour"] = g["water_kwh_day"] / max(len(water_hours), 1)

    df = hourly.copy()
    df["DAY"] = df["DATE"].dt.normalize()
    df["HOUR_OF_DAY"] = df["DATE"].dt.hour

    merged = df.merge(
        g[["DAY", "water_kwh_per_hour"]],
        on="DAY",
        how="left",
    )

    # Base electricity (original usage)
    merged["base_kwh"] = merged["USAGE (kWh)"].astype(float)

    # Water heating component (only in specified hours, from start date)
    merged["water_heat_kwh"] = 0.0
    active = merged["DATE"] >= start
    water_mask = merged["HOUR_OF_DAY"].isin(water_hours)
    merged.loc[active & water_mask, "water_heat_kwh"] = merged.loc[active & water_mask, "water_kwh_per_hour"].fillna(0.0)

    # Total
    merged["total_kwh"] = merged["base_kwh"] + merged["water_heat_kwh"]

    return merged.drop(columns=[
        "DAY", "HOUR_OF_DAY", "water_kwh_per_hour", "USAGE (kWh)"
    ])


if __name__ == "__main__":

    RAW_CSV = Path("../output/pge_usage_2025_260204.csv")
    OUT_CSV = Path("../output/260204_2025_adjusted_kWh.csv")
    PLOT_PNG = Path("../figs/260204_original_vs_adjusted.png")
    PLOT_PNG_HOD = Path("../figs/260204_hour_of_day_profiles.png")
    START_DATE = pd.Timestamp("2025-01-01")

    raw = load_usage(RAW_CSV)
    # Aggregate electric 15-min readings to hourly
    hourly_elec = aggregate_hourly_usage(raw)
    # Compute daily gas distribution
    gas_daily = daily_gas(raw)
    # Apply gas-equivalent to hourly data with component columns
    result = add_gas_equiv_kwh(hourly_elec, gas_daily, START_DATE)

    # Filter to records on/after START_DATE
    filtered = result[result["DATE"] >= START_DATE]
    filtered.to_csv(OUT_CSV, index=False)

    # Audit of added kWh
    total_therms = gas_daily.loc[gas_daily["DAY"] >= START_DATE.normalize(), "gas_therms"].sum()
    expected_water_heat = total_therms * THERM_WATER_TO_KWH
    actual_water_heat = filtered["water_heat_kwh"].sum()

    # === Plots: base vs total ===
    if not filtered.empty:
        sns.set_theme(style="darkgrid", context="talk")

        # Hourly overlay and daily totals overlay
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

        # 1) Hourly time series
        f_long = filtered.melt(id_vars=["DATE"], value_vars=["base_kwh", "water_heat_kwh", "total_kwh"],
                               var_name="series", value_name="kWh")
        sns.lineplot(data=f_long, x="DATE", y="kWh", hue="series", ax=axes[0], linewidth=1)
        axes[0].set_title("Hourly kWh — Base, Water Heat, Total")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("kWh")
        axes[0].legend(title="Series")

        # 2) Daily total time series
        daily = filtered.copy()
        daily["DAY"] = daily["DATE"].dt.normalize()
        daily_tot = (daily.groupby("DAY")[['base_kwh','water_heat_kwh','total_kwh']].sum().reset_index())
        d_long = daily_tot.melt(id_vars=["DAY"], value_vars=["base_kwh", "water_heat_kwh", "total_kwh"],
                                var_name="series", value_name="kWh")
        sns.lineplot(data=d_long, x="DAY", y="kWh", hue="series", ax=axes[1], linewidth=2)
        axes[1].set_title("Daily Total kWh — Base, Water Heat, Total")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("kWh/day")
        axes[1].legend(title="Series")

        # Titles/annotations
        try:
            t0 = pd.to_datetime(filtered['DATE'].min()).date()
            t1 = pd.to_datetime(filtered['DATE'].max()).date()
            fig.suptitle(f"Electricity Usage Components ({t0} → {t1})", y=0.98)
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150, bbox_inches='tight')

        # === Second figure: Hour-of-day profiles (Weekday vs Weekend) ===
        prof = filtered.copy()
        prof["HOUR"] = prof["DATE"].dt.hour
        prof["day_type"] = prof["DATE"].dt.dayofweek.map(lambda d: "Weekend" if d >= 5 else "Weekday")
        prof_avg = (
            prof.groupby(["day_type", "HOUR"])[["base_kwh", "water_heat_kwh", "total_kwh"]]
                .mean()
                .reset_index()
        )
        p_long = prof_avg.melt(id_vars=["day_type", "HOUR"], value_vars=["base_kwh", "water_heat_kwh", "total_kwh"],
                               var_name="series", value_name="kWh")
        g = sns.FacetGrid(p_long, col="day_type", hue="series", height=4, aspect=2, sharey=True)
        g.map_dataframe(sns.lineplot, x="HOUR", y="kWh", linewidth=2)
        g.add_legend(title="Series")
        g.set_axis_labels("Hour of day", "kWh")
        for ax in g.axes.flat:
            ax.set_xticks(range(0, 24))
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Average Hour-of-Day kWh — Weekday vs Weekend")
        g.fig.savefig(PLOT_PNG_HOD, dpi=150, bbox_inches='tight')
    else:
        print("⚠️  No data on/after START_DATE; skipping plots.")

    # Annual summary
    total_base = filtered["base_kwh"].sum()
    total_water = filtered["water_heat_kwh"].sum()
    total_kwh = filtered["total_kwh"].sum()

    print(f"✅  Saved adjusted file for period {START_DATE.date()} onwards → {OUT_CSV.resolve()}")
    print(f"\n📊 Annual Summary:")
    print(f"   Base electricity:      {total_base:,.0f} kWh")
    print(f"   Water heating:         {total_water:,.0f} kWh")
    print(f"   Total:                 {total_kwh:,.0f} kWh")
    print(f"\nℹ️  Audit — total gas therms: {total_therms:.2f}")
    print(f"ℹ️  Audit — water heating kWh (expected): {expected_water_heat:.2f}, (actual): {actual_water_heat:.2f}")

    # Ensure a single show call at the end of the script
    plt.show()
