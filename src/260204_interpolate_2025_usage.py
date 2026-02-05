"""
Simple pass-through for full year 2025 usage data.
No interpolation/projection needed since we have Jan-Dec 2025.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Input/Output ---
USAGE_FP = Path("../input/pge_usage_2025-01-01_to_2025-12-31.csv")
OUT_FP = Path("../output/pge_usage_2025_260204.csv")
FIGS_DIR = Path("../figs/install/")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load usage data ---
usage = pd.read_csv(USAGE_FP)
usage["DATE"] = pd.to_datetime(usage["DATE"])
usage["TYPE"] = usage["TYPE"].str.strip()
usage["USAGE (kWh)"] = pd.to_numeric(usage["USAGE (kWh)"], errors="coerce")

# --- Filter to 2025 only ---
usage_2025 = usage[usage["DATE"].dt.year == 2025].copy()

# --- Separate electric and gas ---
elec = usage_2025[usage_2025["TYPE"].str.contains("Electric", case=False, na=False)].copy()
gas = usage_2025[usage_2025["TYPE"].str.strip().eq("Natural gas usage")].copy()

# --- Add NOTES column ---
elec["NOTES"] = "observed"
gas["NOTES"] = "observed"

# --- Sort by date/time ---
def time_to_minutes(t):
    try:
        parts = str(t).split(":")
        return int(parts[0]) * 60 + int(parts[1]) if len(parts) >= 2 else 0
    except:
        return 0

elec["sort_key"] = elec["START TIME"].apply(time_to_minutes)
elec = elec.sort_values(["DATE", "sort_key"]).drop(columns=["sort_key"])

gas["sort_key"] = gas["START TIME"].apply(time_to_minutes)
gas = gas.sort_values(["DATE", "sort_key"]).drop(columns=["sort_key"])

# --- Combine: electric first, then gas ---
cols = ["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"]
final = pd.concat([elec[cols], gas[cols]], ignore_index=True)

# --- Write output ---
final.to_csv(OUT_FP, index=False)

# --- Monthly aggregation ---
monthly_elec = (
    elec.groupby(elec["DATE"].dt.to_period("M"))["USAGE (kWh)"]
    .sum()
    .rename("electricity_kWh")
)
monthly_gas = (
    gas.groupby(gas["DATE"].dt.to_period("M"))["USAGE (kWh)"]
    .sum()
    .rename("gas_therms")
)

# --- Summary ---
elec_total = elec["USAGE (kWh)"].sum()
gas_total = gas["USAGE (kWh)"].sum()
print(f"Loaded 2025 usage data:")
print(f"  Electric rows: {len(elec):,}")
print(f"  Gas rows: {len(gas):,}")
print(f"  Total electric: {elec_total:,.1f} kWh")
print(f"  Total gas: {gas_total:,.1f} therms")
print(f"Wrote to: {OUT_FP}")

# --- Monthly summary table ---
print("\nMonthly totals:")
for period in monthly_elec.index:
    e = monthly_elec.get(period, 0)
    g = monthly_gas.get(period, 0)
    print(f"  {period}: {e:,.0f} kWh, {g:,.1f} therms")

# --- Plots ---
# Electricity by month
plt.figure(figsize=(10, 5))
monthly_elec.plot(kind="bar", color="steelblue")
plt.title("2025 Monthly Electricity Usage")
plt.xlabel("Month")
plt.ylabel("kWh")
plt.xticks(range(len(monthly_elec)), [p.strftime("%b") for p in monthly_elec.index], rotation=45)
plt.tight_layout()
plt.savefig(FIGS_DIR / "260204_monthly_electricity.png")
print(f"Saved: {FIGS_DIR / '260204_monthly_electricity.png'}")

# Gas by month
plt.figure(figsize=(10, 5))
monthly_gas.plot(kind="bar", color="darkorange")
plt.title("2025 Monthly Gas Usage")
plt.xlabel("Month")
plt.ylabel("therms")
plt.xticks(range(len(monthly_gas)), [p.strftime("%b") for p in monthly_gas.index], rotation=45)
plt.tight_layout()
plt.savefig(FIGS_DIR / "260204_monthly_gas.png")
print(f"Saved: {FIGS_DIR / '260204_monthly_gas.png'}")

plt.show()
