import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# CONFIGURATION
FILE_PATH      = '../input/pge_usage_2024-08-01_to_2025-07-31.csv'
BASELINE_START = '2025-01-01'
BASELINE_END   = '2025-07-31'
ETA_FURNACE    = 0.95      # furnace efficiency
COP_HP         = 2.78      # heat-pump coefficient of performance
# --------------------------------------------------------------------------------

# 1. Load & prep
df = pd.read_csv(FILE_PATH)
df['DATE'] = pd.to_datetime(df['DATE'])

df_elec = df[df['TYPE'] == 'Electric usage'].copy()
df_gas  = df[df['TYPE'] == 'Natural gas usage'].copy()
df_gas  = df_gas.rename(columns={'USAGE (kWh)': 'usage_therms'})   # mis-label in raw

# 2. Baseline non-heating gas
mask_baseline     = (df_gas['DATE'] >= BASELINE_START) & (df_gas['DATE'] <= BASELINE_END)
baseline_daily_th = df_gas.loc[mask_baseline, 'usage_therms'].mean()

# 3. Heating therms → kWh before Jan 1 2025
CONV = 29.3 / (ETA_FURNACE * COP_HP)  # kWh per therm of delivered heat
df_gas['heating_therms'] = np.where(
    df_gas['DATE'] < BASELINE_START,
    np.maximum(0, df_gas['usage_therms'] - baseline_daily_th),
    0.0
)
df_gas['heating_kwh'] = df_gas['heating_therms'] * CONV

# 4. Monthly aggregation
df_elec['month'] = df_elec['DATE'].dt.to_period('M')
df_gas['month']  = df_gas['DATE'].dt.to_period('M')


monthly = (
    df_elec.groupby('month')['USAGE (kWh)'].sum().to_frame('elec_kwh')
    # ⬇︎ INCLUDE BOTH heating_kwh **and** usage_therms here
    .join(df_gas.groupby('month')[['heating_kwh', 'usage_therms']].sum(),
          how='left')
    .fillna(0.0)
)
monthly['elec_adj_kwh'] = monthly['elec_kwh'] + monthly['heating_kwh']

print(monthly)

# 5. Plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 5))

x = monthly.index.to_timestamp()

# Electricity lines (left axis)
ax1.plot(x, monthly['elec_kwh'],  marker='o', label='Actual electricity (kWh)')
ax1.plot(x, monthly['elec_adj_kwh'], marker='o',
         label='Adjusted electricity incl. heating (kWh)')

ax1.set_xlabel('Month')
ax1.set_ylabel('kWh')
ax1.grid(True)

# Gas therms (right axis)
ax2 = ax1.twinx()
ax2.plot(x, monthly['usage_therms'], marker='s', linestyle='--',
         label='Total gas usage (therms)')
ax2.set_ylabel('Therms')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Monthly Energy Usage: Electricity vs Gas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()