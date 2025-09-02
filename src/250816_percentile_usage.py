import pandas as pd
import matplotlib.pyplot as plt

file_path = '../input/pge_electric_usage_interval_data_Service 1_1_2025-01-01_to_2025-08-14.csv'

# Reload CSV but skip metadata rows before the actual header line
df = pd.read_csv(file_path, skiprows=6)

# Clean up column names
df.columns = [c.strip() for c in df.columns]

# Ensure DATE is datetime
df["DATE"] = pd.to_datetime(df["DATE"])

# Group by DATE to get daily totals
daily_usage = df.groupby("DATE")["USAGE (kWh)"].sum().reset_index()

# Add month column
daily_usage["Month"] = daily_usage["DATE"].dt.to_period("M")

# Calculate 90th percentile per month
percentiles = daily_usage.groupby("Month")["USAGE (kWh)"].quantile(0.9).reset_index()
percentiles.rename(columns={"USAGE (kWh)": "90th Percentile (kWh/day)"}, inplace=True)
print(percentiles)


a = 1

plt.figure()
for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    perc = daily_usage.groupby("Month")["USAGE (kWh)"].quantile(p)
    plt.plot([str(m) for m in perc.index], perc.values, label=p)

plt.legend()
plt.show()
