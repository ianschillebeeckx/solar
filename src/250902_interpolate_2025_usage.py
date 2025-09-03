import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Inputs (paths must match your uploaded files) ---
GAS_2024_FP = Path("../input/pge_natural_gas_usage_interval_data_Service 2_2_2024-01-01_to_2024-12-31.csv")
USAGE_2025_FP = Path("../input/pge_usage_2024-09-01_to_2025-08-31.csv")

# --- Load 2024 gas (skip PG&E metadata header) ---
gas_2024 = pd.read_csv(GAS_2024_FP, skiprows=6)
gas_2024["DATE"] = pd.to_datetime(gas_2024["DATE"])
gas_2024["USAGE (therms)"] = pd.to_numeric(gas_2024["USAGE (therms)"], errors="coerce")
monthly_gas_2024 = (
    gas_2024.groupby(gas_2024["DATE"].dt.to_period("M"))["USAGE (therms)"]
    .sum()
    .rename("gas_therms_2024")
    .astype(float)
)

# --- Preserve actual 2024 monthly therms and build an interpolated version for May/June ---
monthly_gas_2024_actual = monthly_gas_2024.copy()
monthly_gas_2024_sanitized = monthly_gas_2024.copy()

# Linear interpolation for May (5) and June (6) between April (4) and July (7)
p_apr, p_may, p_jun, p_jul = [pd.Period(f"{2024}-{m:02d}", freq="M") for m in (4,5,6,7)]
apr_val = monthly_gas_2024_actual.get(p_apr, np.nan)
jul_val = monthly_gas_2024_actual.get(p_jul, np.nan)
if not (np.isnan(apr_val) or np.isnan(jul_val)):
    x_known = np.array([4, 7], dtype=float)
    y_known = np.array([apr_val, jul_val], dtype=float)
    x_target = np.array([5, 6], dtype=float)
    y_interp = np.interp(x_target, x_known, y_known)
    monthly_gas_2024_sanitized.loc[p_may] = y_interp[0]
    monthly_gas_2024_sanitized.loc[p_jun] = y_interp[1]

# choose which series to use for mapping/plots (sanitized)
monthly_gas_2024_used = monthly_gas_2024_sanitized

# --- Load combined usage (electric + possibly gas) for Sep 2024 – Aug 2025 ---
usage = pd.read_csv(USAGE_2025_FP)
usage["DATE"] = pd.to_datetime(usage["DATE"])
usage["TYPE"] = usage["TYPE"].str.strip()

# Electricity aggregation
if "USAGE (kWh)" in usage:
    usage["USAGE (kWh)"] = pd.to_numeric(usage["USAGE (kWh)"], errors="coerce")
    monthly_elec = (
        usage.loc[usage["TYPE"].str.contains("Electric", case=False, na=False)]
        .groupby(usage["DATE"].dt.to_period("M"))["USAGE (kWh)"]
        .sum()
        .rename("electricity_kWh")
        .astype(float)
    )
else:
    monthly_elec = pd.Series(dtype=float, name="electricity_kWh")

# Gas aggregation (PG&E CSV encodes therms in 'USAGE (kWh)' for 'Natural gas usage' rows)
usage["USAGE (kWh)"] = pd.to_numeric(usage["USAGE (kWh)"], errors="coerce")
monthly_gas_2025_actual = (
    usage.loc[usage["TYPE"].str.strip().eq("Natural gas usage")]
    .groupby(usage["DATE"].dt.to_period("M"))["USAGE (kWh)"]
    .sum()
    .rename("gas_therms_2025_actual")
    .astype(float)
)


# --- Build 2024 mapping: for each late month (Sep–Dec), find the closest early month (Jan–Aug) by |therms difference| ---
early_months_2024 = [pd.Period(f"2024-{m:02d}", freq="M") for m in range(1, 9)]
late_months_2024  = [pd.Period(f"2024-{m:02d}", freq="M") for m in range(9, 13)]

mapping_2024 = {}
for tgt in late_months_2024:
    tgt_val = monthly_gas_2024_used.get(tgt, np.nan)
    diffs = {src: abs(monthly_gas_2024_used.get(src, np.nan) - tgt_val) for src in early_months_2024}
    diffs = {k: v for k, v in diffs.items() if not np.isnan(v)}
    mapping_2024[tgt] = (min(diffs, key=diffs.get) if diffs else None)

# --- Apply mapping to project 2025 Sep–Dec by copying 2025 Jan–Aug values from the mapped months ---
months_2025_actual = [pd.Period(f"{2025}-{m:02d}", freq="M") for m in range(1, 9)]

elec_2025_actual = monthly_elec[monthly_elec.index.isin(months_2025_actual)]
gas_2025_actual  = monthly_gas_2025_actual[monthly_gas_2025_actual.index.isin(months_2025_actual)]

targets_2025 = [pd.Period(f"2025-{m:02d}", freq="M") for m in range(9, 13)]
proj_elec_2025 = {}
proj_gas_2025 = {}
map_2024_to_2025 = {}

for tgt_2025 in targets_2025:
    # use the 2024 mapping for the same calendar month
    tgt_2024 = pd.Period(f"2024-{tgt_2025.month:02d}", freq="M")
    src_2024 = mapping_2024.get(tgt_2024)
    if src_2024 is None:
        map_2024_to_2025[tgt_2025] = None
        proj_elec_2025[tgt_2025] = np.nan
        proj_gas_2025[tgt_2025] = np.nan
        continue
    src_2025 = pd.Period(f"2025-{src_2024.month:02d}", freq="M")
    map_2024_to_2025[tgt_2025] = src_2025
    proj_elec_2025[tgt_2025] = elec_2025_actual.get(src_2025, np.nan)
    proj_gas_2025[tgt_2025] = gas_2025_actual.get(src_2025, np.nan)

# --- Assemble final table ---
all_months_2025 = [pd.Period(f"2025-{m:02d}", freq="M") for m in range(1, 13)]

elec_2025_full = pd.Series(index=all_months_2025, dtype=float, name="electricity_kWh")
elec_2025_full.update(elec_2025_actual)
elec_2025_full.update(pd.Series(proj_elec_2025))

gas_2025_full = pd.Series(index=all_months_2025, dtype=float, name="gas_therms")
gas_2025_full.update(gas_2025_actual)
gas_2025_full.update(pd.Series(proj_gas_2025))

result_2025 = pd.concat([elec_2025_full, gas_2025_full], axis=1)
result_2025.index = result_2025.index.to_timestamp()

# --- Print results ---
# Report actual vs interpolated for May/June (therms)
print("2024 May/June actual vs interpolated (therms):")
print("  May 2024: actual = {:.2f}, interpolated = {:.2f}".format(
    monthly_gas_2024_actual.get(pd.Period("2024-05", freq="M"), np.nan),
    monthly_gas_2024_used.get(pd.Period("2024-05", freq="M"), np.nan)
))
print("  Jun 2024: actual = {:.2f}, interpolated = {:.2f}".format(
    monthly_gas_2024_actual.get(pd.Period("2024-06", freq="M"), np.nan),
    monthly_gas_2024_used.get(pd.Period("2024-06", freq="M"), np.nan)
))
print("2024 late→early mapping (by therm similarity):")
for k, v in mapping_2024.items():
    print(f"  {k.strftime('%Y-%m')} -> {v.strftime('%Y-%m') if v is not None else 'None'}")

# =============================
# Build interval-level 2025 series (15-min electric, daily gas) via shape→scale
# =============================
from calendar import monthrange

def tod_bin_from_time_str(t):
    s = str(t)
    parts = s.split(":")
    try:
        h = int(parts[0])
    except Exception:
        h = 0
    try:
        m = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        m = 0
    return (h * 60 + m) // 15

# Convenience columns for profiling
if "START TIME" in usage.columns:
    usage["start_bin"] = usage["START TIME"].apply(tod_bin_from_time_str)
else:
    usage["start_bin"] = np.nan
usage["dow"] = usage["DATE"].dt.dayofweek

# Electricity profiling (DOW × 15-min)

def build_electric_profile(src_period):
    df = usage[(usage["TYPE"].str.contains("Electric", case=False, na=False)) & (usage["DATE"].dt.to_period("M") == src_period)]
    if df.empty:
        return None, None
    grp = df.groupby(["dow", "start_bin"], as_index=False)["USAGE (kWh)"].mean().rename(columns={"USAGE (kWh)": "kwh_mean"})
    bin_means = grp.groupby("start_bin", as_index=False)["kwh_mean"].mean().rename(columns={"kwh_mean": "kwh_mean_bin"})
    return grp, bin_means


def gen_electric_15min_month(target_period, src_period, month_total_kwh):
    start = pd.Timestamp(target_period.start_time)
    end = pd.Timestamp(target_period.end_time)
    rng = pd.date_range(start, end, freq="15T", inclusive="left")
    out = pd.DataFrame({"DATETIME": rng})
    out["DATE"] = out["DATETIME"].dt.date.astype(str)
    out["START TIME"] = out["DATETIME"].dt.strftime("%H:%M")
    out["END TIME"] = (out["DATETIME"] + pd.Timedelta(minutes=14)).dt.strftime("%H:%M")
    out["dow"] = out["DATETIME"].dt.dayofweek
    out["start_bin"] = (out["DATETIME"].dt.hour * 60 + out["DATETIME"].dt.minute) // 15

    prof, bin_means = build_electric_profile(src_period)
    if prof is None:
        out["baseline"] = 1.0
    else:
        out = out.merge(prof, on=["dow", "start_bin"], how="left")
        out = out.merge(bin_means, on=["start_bin"], how="left")
        out["baseline"] = out["kwh_mean"].fillna(out["kwh_mean_bin"]).fillna(1.0)

    s = out["baseline"].sum()
    scale = (month_total_kwh / s) if s and not np.isnan(month_total_kwh) else 0.0
    out["USAGE (kWh)"] = out["baseline"] * scale
    out["TYPE"] = "Electric usage"
    out["NOTES"] = f"projected from {src_period.strftime('%b %Y')}"
    out["COST"] = ""
    return out[["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"]]

# Gas profiling (DOW)

def build_gas_dow_profile(src_period):
    df = usage[(usage["TYPE"].str.strip().eq("Natural gas usage")) & (usage["DATE"].dt.to_period("M") == src_period)]
    if df.empty:
        return None
    daily = df.groupby(df["DATE"].dt.date)["USAGE (kWh)"]
    prof = daily.sum().groupby(pd.to_datetime(daily.sum().index).dayofweek).mean()
    prof = prof.reindex(range(7)).fillna(prof.mean())
    return prof


def gen_gas_daily_month(target_period, src_period, month_total_therms):
    days_in_month = monthrange(target_period.year, target_period.month)[1]
    dates = pd.date_range(target_period.start_time, periods=days_in_month, freq="D")
    out = pd.DataFrame({"DATE": dates})
    out["dow"] = out["DATE"].dt.dayofweek
    prof = build_gas_dow_profile(src_period)
    if prof is None:
        out["baseline"] = 1.0
    else:
        out["baseline"] = out["dow"].map(prof)
    s = out["baseline"].sum()
    scale = (month_total_therms / s) if s and not np.isnan(month_total_therms) else 0.0
    out["USAGE (kWh)"] = out["baseline"] * scale
    out["TYPE"] = "Natural gas usage"
    out["DATE"] = out["DATE"].dt.date
    out["START TIME"] = "0:00"
    out["END TIME"] = "23:59"
    out["COST"] = ""
    out["NOTES"] = f"projected from {src_period.strftime('%b %Y')}"
    return out[["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"]]

# Observed 2025 rows from raw file
observed_elec_2025 = (
    usage[(usage["TYPE"].str.contains("Electric", case=False, na=False)) & (usage["DATE"].dt.year == 2025)]
    .assign(NOTES="observed")
    [["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"]]
)
observed_gas_2025 = (
    usage[(usage["TYPE"].str.strip().eq("Natural gas usage")) & (usage["DATE"].dt.year == 2025)]
    .assign(NOTES="observed")
    [["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"]]
)

# Generate projections for Sep–Dec 2025
proj_elec_rows, proj_gas_rows = [], []
for tgt_2025 in targets_2025:
    src_2025 = map_2024_to_2025.get(tgt_2025)
    if src_2025 is None:
        continue
    month_total_kwh = result_2025.loc[tgt_2025.to_timestamp(), "electricity_kWh"]
    month_total_therms = result_2025.loc[tgt_2025.to_timestamp(), "gas_therms"]
    proj_elec_rows.append(gen_electric_15min_month(tgt_2025, src_2025, month_total_kwh))
    proj_gas_rows.append(gen_gas_daily_month(tgt_2025, src_2025, month_total_therms))

proj_elec_2025 = pd.concat(proj_elec_rows, ignore_index=True) if proj_elec_rows else pd.DataFrame()
proj_gas_2025  = pd.concat(proj_gas_rows,  ignore_index=True) if proj_gas_rows else pd.DataFrame()

# Combine observed and projected for 2025
final_elec_2025 = pd.concat([observed_elec_2025, proj_elec_2025], ignore_index=True)
final_gas_2025  = pd.concat([observed_gas_2025,  proj_gas_2025],  ignore_index=True)

# Helper formatters to match PG&E file style
fmt_date = lambda s: pd.to_datetime(s).strftime("%-m/%-d/%Y")

def fmt_time(s):
    s = str(s)
    if ":" not in s:
        return s
    h, m = s.split(":")
    try:
        return f"{int(h)}:{int(m):02d}"
    except Exception:
        return s

# Sort each segment internally, then append gas after electric
for df in (final_elec_2025, final_gas_2025):
    df["DATE_ts"] = pd.to_datetime(df["DATE"])  # ensure
    df["start_bin"] = df["START TIME"].apply(tod_bin_from_time_str)
    df.sort_values(["DATE_ts", "start_bin"], inplace=True)
    # format
    df["DATE"] = df["DATE_ts"].apply(fmt_date)
    df["START TIME"] = df["START TIME"].apply(fmt_time)
    df["END TIME"] = df["END TIME"].apply(fmt_time)
    df.drop(columns=["DATE_ts", "start_bin"], inplace=True)

# Exact column order from the source file
ecols = ["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"]
final_elec_2025 = final_elec_2025[ecols]
final_gas_2025  = final_gas_2025[ecols]

# Append gas after electricity (no interleaving)
final_all_2025 = pd.concat([final_elec_2025, final_gas_2025], ignore_index=True)

# --- Export full 2025 interval dataset to CSV ---
OUT_FP = Path("../output/pge_usage_2025_projected.csv")
final_all_2025.to_csv(OUT_FP, index=False)
print(f"Wrote final 2025 interval CSV to: {OUT_FP}")

# --- Sanity check (requested): compare synthesized monthly total for target month against its TEMPLATE month total ---
print("Sanity check vs template months (Electricity & Gas):")
fa = final_all_2025.copy()
fa["DATE_ts"] = pd.to_datetime(fa["DATE"])  # ensure
fa["period"] = fa["DATE_ts"].dt.to_period("M")

for tgt_2025 in targets_2025:
    src_2025 = map_2024_to_2025.get(tgt_2025)
    if src_2025 is None:
        continue
    # synthesized from intervals
    elec_calc = fa.loc[(fa["TYPE"].eq("Electric usage")) & (fa["period"].eq(tgt_2025)), "USAGE (kWh)"].sum()
    gas_calc  = fa.loc[(fa["TYPE"].eq("Natural gas usage")) & (fa["period"].eq(tgt_2025)), "USAGE (kWh)"].sum()
    # template (expected) from source month totals actually observed in 2025
    elec_exp = monthly_elec.get(src_2025, np.nan)
    gas_exp  = monthly_gas_2025_actual.get(src_2025, np.nan)
    elec_diff = elec_calc - elec_exp if not np.isnan(elec_exp) else np.nan
    gas_diff  = gas_calc  - gas_exp  if not np.isnan(gas_exp)  else np.nan
    print(
        f"  {tgt_2025.strftime('%Y-%m')} (src {src_2025.strftime('%Y-%m')}):"
        f"    Electric  expected={elec_exp:.2f}  calculated={elec_calc:.2f}  diff={elec_diff:.6f}"
        f"    Gas       expected={gas_exp:.2f}   calculated={gas_calc:.2f}   diff={gas_diff:.6f}"
    )

# --- Visualization: 2024 mapping + 2025 series ---
plt.figure()
months_2024 = [pd.Period(f"2024-{m:02d}", freq="M") for m in range(1, 13)]
x_idx = np.arange(1, 13)
y_vals = [monthly_gas_2024_used.get(m, np.nan) for m in months_2024]
plt.plot(x_idx, y_vals, marker="o", label="Interpolated (used for mapping)")
# overlay actual May/June values for comparison
may_y = monthly_gas_2024_actual.get(pd.Period("2024-05", freq="M"), np.nan)
jun_y = monthly_gas_2024_actual.get(pd.Period("2024-06", freq="M"), np.nan)
plt.scatter([5, 6], [may_y, jun_y], marker="x", s=80, label="Actual May/Jun")
plt.legend()
plt.xticks(x_idx, [m.strftime("%b") for m in months_2024], rotation=45)
plt.title("2024 Gas (therms) with Late→Early Mapping Lines")
plt.xlabel("Month (2024)")
plt.ylabel("therms")

# Draw horizontal lines from early (Jan–Aug) to late (Sep–Dec) at the early-month value
early_months_2024 = [pd.Period(f"2024-{m:02d}", freq="M") for m in range(1, 9)]
late_months_2024  = [pd.Period(f"2024-{m:02d}", freq="M") for m in range(9, 13)]
for late in late_months_2024:
    early = mapping_2024.get(late)
    if early is None:
        continue
    y = monthly_gas_2024_used.get(early, np.nan)
    if np.isnan(y):
        continue
    plt.hlines(y=y, xmin=early.month, xmax=late.month, linestyles="dashed")
    mid_x = (early.month + late.month) / 2.0
    plt.text(mid_x, y, f"{late.strftime('%b')} ← {early.strftime('%b')}", ha="center", va="bottom")

# 2) 2025 Electricity
plt.figure()
result_2025["electricity_kWh"].plot(marker="o")
plt.title("2025 Electricity (kWh): Actual Jan–Aug; Projected Sep–Dec via Month Mapping")
plt.xlabel("Month")
plt.ylabel("kWh")
plt.xticks(rotation=45)
plt.tight_layout()

# 3) 2025 Gas
plt.figure()
result_2025["gas_therms"].plot(marker="o")
plt.title("2025 Gas (therms): Actual Jan–Aug; Projected Sep–Dec via Month Mapping")
plt.xlabel("Month")
plt.ylabel("therms")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
