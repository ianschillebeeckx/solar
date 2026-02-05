import pandas as pd
import numpy as np
from pathlib import Path

# === Config ===
FURNACE_EFFICIENCY = 0.90  # e.g., 85% efficient gas furnace
HEAT_PUMP_COP = 3.0  # Coefficient of performance of heat pump


# === Load and preprocess functions ===
def load_gas(file_path):
    df = pd.read_csv(file_path, skiprows=6)
    df = df.rename(columns={"USAGE (therms)": "therms", "DATE": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["therms"] = pd.to_numeric(df["therms"], errors="coerce")

    # Smooth 0,0,0,1 patterns
    corrected = df["therms"].copy()
    i = 0
    while i < len(corrected):
        if corrected[i] == 0:
            start = i
            while i < len(corrected) and corrected[i] == 0:
                i += 1
            if i < len(corrected) and corrected[i] > 0:
                total = corrected[i]
                length = i - start + 1
                distributed = total / length
                corrected.iloc[start:i + 1] = distributed
            i += 1
        else:
            i += 1
    df["smoothed_therms"] = corrected
    return df[["date", "smoothed_therms"]]


def load_electricity(file_path):
    df = pd.read_csv(file_path, skiprows=6)
    df = df.rename(columns={"USAGE (kWh)": "kWh", "DATE": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce")
    return df.groupby("date")["kWh"].sum().reset_index()


# === Load data ===
data_dir = Path("../input/")

gas_2024 = load_gas(data_dir / "pge_natural_gas_usage_interval_data_Service 2_2_2024-01-01_to_2024-12-31.csv")
gas_2025 = load_gas(data_dir / "pge_natural_gas_usage_interval_data_Service 2_2_2025-01-01_to_2025-08-14.csv")
elec_2024 = load_electricity(data_dir / "pge_electric_usage_interval_data_Service 1_1_2024-01-01_to_2024-12-31.csv")
elec_2025 = load_electricity(data_dir / "pge_electric_usage_interval_data_Service 1_1_2025-01-01_to_2025-08-14.csv")

# === Step 1: Estimate baseline gas usage from 2025 (non-HVAC) ===
gas_2025_baseline = gas_2025.set_index("date").resample("D").mean().fillna(method="ffill")["smoothed_therms"]

# === Step 2: Subtract baseline from 2024 gas to isolate HVAC usage ===
gas_2024 = gas_2024.set_index("date").resample("D").sum()
gas_2024["baseline"] = gas_2025_baseline.reindex(gas_2024.index, method="nearest")
gas_2024["hvac_therms"] = (gas_2024["smoothed_therms"] - gas_2024["baseline"]).clip(lower=0)

# === Step 3: Convert post-Aug 15 2024 HVAC therms to kWh ===
conversion_factor = 29.3 * FURNACE_EFFICIENCY / HEAT_PUMP_COP
hvac_kwh_2024 = gas_2024.loc["2024-08-15":, "hvac_therms"] * conversion_factor

# === Step 4: Add to actual electricity usage to estimate 2025 ===
elec_2024 = elec_2024.set_index("date").reindex(gas_2024.index, fill_value=0)

elec_2025_est = elec_2024["kWh"].copy()
elec_2025_est.loc["2024-08-15":] += hvac_kwh_2024

elec_2025_est = elec_2025_est.reset_index().rename(columns={"kWh": "estimated_2025_kWh"})

# === Result: Daily electricity estimate for 2025 ===
elec_2025_est
