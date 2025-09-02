from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === Conversion constants ===
# Keep the blended constant for reference, but we now allocate by end-use and hours of day.
THERM_TO_DELIVERED_KWH = 29.3 * (0.5 * 0.38 / 0.87 + 0.5 * 0.90 / 3.9)  # ≈ 9.8
THERM_COOKING_TO_KWH   = 29.3 * (0.5 * 0.38 / 0.87)  # total daily kWh from the cooking half
THERM_WATER_TO_KWH     = 29.3 * (0.5 * 0.90 / 3.9)  # total daily kWh from the water-heating half

# Schedules (24h clock). You asked for water at 11/12/13 and cooking at 17/18.
WATER_HEAT_HOURS   = (11, 12, 13)   # three hours at midday
COOKING_HOURS      = (17, 18)       # 5pm and 6pm

# EV charging profile defaults
# Assumption: "between 10a and 3p" → hours 10,11,12,13,14 (5 hours). Adjust as needed.
EV_CHARGE_HOURS    = (10, 11, 12, 13, 14)
# Pandas dayofweek: Mon=0,...,Sun=6 → Sunday, Wednesday, Friday, Saturday
EV_CHARGE_DAYS     = (6, 2, 4, 5)
EV_WEEKLY_KWH      = 59.0


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
    cooking_hours: tuple[int, ...] = COOKING_HOURS,
) -> pd.DataFrame:
    """Return hourly DataFrame with original + gas-adjusted kWh using end-use hour blocks.

    Allocation
    ----------
    * Water heating: split evenly across ``water_hours`` (default 11:00, 12:00, 13:00).
    * Cooking: split evenly across ``cooking_hours`` (default 17:00 and 18:00).

    Each day's distributed therms are converted to kWh by end-use (using
    ``THERM_WATER_TO_KWH`` and ``THERM_COOKING_TO_KWH``) and then placed only into
    the specified hours for that same day.
    """
    # Compute per-day end-use kWh and per-hour allocations
    g = gas_daily.copy()
    g = g.assign(
        water_kwh_day=lambda x: x["gas_therms"] * THERM_WATER_TO_KWH,
        cook_kwh_day=lambda x: x["gas_therms"] * THERM_COOKING_TO_KWH,
    )
    g["water_kwh_per_hour"] = g["water_kwh_day"] / max(len(water_hours), 1)
    g["cook_kwh_per_hour"] = g["cook_kwh_day"] / max(len(cooking_hours), 1)

    df = hourly.copy()
    df["DAY"] = df["DATE"].dt.normalize()
    df["HOUR_OF_DAY"] = df["DATE"].dt.hour

    merged = df.merge(
        g[["DAY", "water_kwh_per_hour", "cook_kwh_per_hour"]],
        on="DAY",
        how="left",
    )

    # Preserve original usage
    merged["original_kwh"] = merged["USAGE (kWh)"].astype(float)
    merged["adjusted_kwh"] = merged["original_kwh"]

    # Apply from start date only
    active = merged["DATE"] >= start

    water_mask = merged["HOUR_OF_DAY"].isin(water_hours)
    cook_mask = merged["HOUR_OF_DAY"].isin(cooking_hours)

    merged.loc[active & water_mask, "adjusted_kwh"] += merged.loc[active & water_mask, "water_kwh_per_hour"].fillna(0.0)
    merged.loc[active & cook_mask, "adjusted_kwh"] += merged.loc[active & cook_mask, "cook_kwh_per_hour"].fillna(0.0)

    return merged.drop(columns=[
        "DAY", "HOUR_OF_DAY", "water_kwh_per_hour", "cook_kwh_per_hour"
    ])


def add_ev_weekly_charging(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    weekly_kwh: float = EV_WEEKLY_KWH,
    charge_days: tuple[int, ...] = EV_CHARGE_DAYS,
    charge_hours: tuple[int, ...] = EV_CHARGE_HOURS,
    week_anchor: str = "W-SUN",
) -> pd.DataFrame:
    """Add EV charging load to ``adjusted_kwh`` after gas end-use adjustments.

    Parameters
    ----------
    hourly : DataFrame
        Must contain columns ["DATE", "USAGE (kWh)", "original_kwh", "adjusted_kwh"].
    start : Timestamp
        Start date for applying EV charging logic.
    weekly_kwh : float
        Total EV charging energy *per week* to allocate.
    charge_days : tuple[int]
        Day-of-week integers (Mon=0..Sun=6) to receive charging.
    charge_hours : tuple[int]
        Hours-of-day (0..23) to receive charging each eligible day.
    week_anchor : str
        Pandas weekly period alias; default groups weeks ending on Sunday ("W-SUN").

    Notes
    -----
    For each week (per ``week_anchor``), we split ``weekly_kwh`` evenly across the
    eligible hours in that week: (#days in ``charge_days`` within the week) ×
    (# ``charge_hours``). This ensures exactly ``weekly_kwh`` per week when all
    eligible hours exist, and scales gracefully for partial weeks.
    """
    df = hourly.copy()
    # If someone calls this without prior gas step, ensure adjusted exists
    if "adjusted_kwh" not in df.columns:
        df["original_kwh"] = df["USAGE (kWh)"]
        df["adjusted_kwh"] = df["USAGE (kWh)"]

    mask_start = df["DATE"] >= start
    if not mask_start.any():
        return df

    sub = df.loc[mask_start, ["DATE"]].copy()
    sub["DAY"] = sub["DATE"].dt.normalize()
    sub["HOUR"] = sub["DATE"].dt.hour
    sub["DOW"] = sub["DATE"].dt.dayofweek
    sub["WEEK"] = sub["DAY"].dt.to_period(week_anchor)

    eligible = (sub["DOW"].isin(charge_days)) & (sub["HOUR"].isin(charge_hours))

    # hours per week that will receive charging
    hours_per_week = (
        sub.assign(eligible=eligible)
           .groupby("WEEK")["eligible"].sum()
           .replace(0, np.nan)  # avoid div-by-zero; yields NaN per-hour which we'll drop
    )
    per_hour = (weekly_kwh / hours_per_week).to_dict()  # map Period -> kWh per eligible hour

    # Compute additions
    add = np.zeros(len(sub), dtype=float)
    idx_eligible = eligible.values
    weeks = sub.loc[idx_eligible, "WEEK"].values
    if len(weeks) > 0:
        # vectorized map
        per_hour_vec = np.array([per_hour.get(w, 0.0) for w in weeks], dtype=float)
        add[idx_eligible] = per_hour_vec

    # Apply back to main df
    df.loc[mask_start, "adjusted_kwh"] = df.loc[mask_start, "adjusted_kwh"].to_numpy() + add

    return df


if __name__ == "__main__":

    RAW_CSV = Path("../input/pge_usage_2024-08-01_to_2025-07-31.csv")
    OUT_CSV = Path("../output/250901_2025_adjusted_kWh_enduse_profiled.csv")
    PLOT_PNG = Path("../figs/250901_original_vs_adjusted.png")
    PLOT_PNG_HOD = Path("../figs/250901_hour_of_day_profiles.png")
    START_DATE = pd.Timestamp("2025-01-01")

    raw = load_usage(RAW_CSV)
    # Aggregate electric 15-min readings to hourly
    hourly_elec = aggregate_hourly_usage(raw)
    # Compute daily gas distribution
    gas_daily = daily_gas(raw)
    # Apply gas-equivalent to hourly data with end-use hour blocks
    after_gas = add_gas_equiv_kwh(hourly_elec, gas_daily, START_DATE)

    # Add EV charging after gas adjustments
    with_ev = add_ev_weekly_charging(
        after_gas,
        START_DATE,
        weekly_kwh=EV_WEEKLY_KWH,
        charge_days=EV_CHARGE_DAYS,
        charge_hours=EV_CHARGE_HOURS,
    )
    # Filter to records on/after START_DATE
    filtered = with_ev[with_ev["DATE"] >= START_DATE]
    filtered.to_csv(OUT_CSV, index=False)

    # Optional: simple audit of added kWh
    total_therms = gas_daily.loc[gas_daily["DAY"] >= START_DATE.normalize(), "gas_therms"].sum()
    expected_gas_added = total_therms * (THERM_COOKING_TO_KWH + THERM_WATER_TO_KWH)

    # Actual gas-only added (compare after_gas vs original)
    mask_after = after_gas["DATE"] >= START_DATE
    actual_gas_added = (after_gas.loc[mask_after, "adjusted_kwh"] - after_gas.loc[mask_after, "original_kwh"]).sum()

    # EV added (compare with_ev vs after_gas)
    idx_common = with_ev.index
    merged_tmp = after_gas.reindex(idx_common)
    mask_with_ev = with_ev["DATE"] >= START_DATE
    ev_added = (with_ev.loc[mask_with_ev, "adjusted_kwh"] - merged_tmp.loc[mask_with_ev, "adjusted_kwh"]).sum()

    total_added_in_file = (filtered["adjusted_kwh"] - filtered["original_kwh"]).sum()

    # === Plots: original vs adjusted ===
    if not filtered.empty:
        sns.set_theme(style="darkgrid", context="talk")

        # Hourly overlay and daily totals overlay
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

        # 1) Hourly time series
        f_long = filtered.melt(id_vars=["DATE"], value_vars=["original_kwh", "adjusted_kwh"],
                               var_name="series", value_name="kWh")
        sns.lineplot(data=f_long, x="DATE", y="kWh", hue="series", ax=axes[0], linewidth=1)
        axes[0].set_title("Hourly kWh — Original vs Adjusted")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("kWh")
        axes[0].legend(title="Series")

        # 2) Daily total time series
        daily = filtered.copy()
        daily["DAY"] = daily["DATE"].dt.normalize()
        daily_tot = (daily.groupby("DAY")[['original_kwh','adjusted_kwh']].sum().reset_index())
        d_long = daily_tot.melt(id_vars=["DAY"], value_vars=["original_kwh", "adjusted_kwh"],
                                var_name="series", value_name="kWh")
        sns.lineplot(data=d_long, x="DAY", y="kWh", hue="series", ax=axes[1], linewidth=2)
        axes[1].set_title("Daily Total kWh — Original vs Adjusted")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("kWh/day")
        axes[1].legend(title="Series")

        # Titles/annotations
        try:
            t0 = pd.to_datetime(filtered['DATE'].min()).date()
            t1 = pd.to_datetime(filtered['DATE'].max()).date()
            fig.suptitle(f"Original vs Adjusted Electricity Usage({t0} → {t1})", y=0.98)
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150, bbox_inches='tight')

        # === Second figure: Hour-of-day profiles (Weekday vs Weekend) ===
        prof = filtered.copy()
        prof["HOUR"] = prof["DATE"].dt.hour
        prof["day_type"] = prof["DATE"].dt.dayofweek.map(lambda d: "Weekend" if d >= 5 else "Weekday")
        prof_avg = (
            prof.groupby(["day_type", "HOUR"])[["original_kwh", "adjusted_kwh"]]
                .mean()
                .reset_index()
        )
        p_long = prof_avg.melt(id_vars=["day_type", "HOUR"], value_vars=["original_kwh", "adjusted_kwh"],
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

    print(f"✅  Saved adjusted file for period {START_DATE.date()} onwards → {OUT_CSV.resolve()}")
    print(f"ℹ️  Audit — expected gas added kWh: {expected_gas_added:.2f}, actual gas added: {actual_gas_added:.2f}")
    print(f"ℹ️  Audit — EV added kWh in file: {ev_added:.2f} (weekly target: {EV_WEEKLY_KWH:.2f} per week)")
    print(f"ℹ️  Audit — total added in file (gas+EV): {total_added_in_file:.2f}")

    # Ensure a single show call at the end of the script
    plt.show()
