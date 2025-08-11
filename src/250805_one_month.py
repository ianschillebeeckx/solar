import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file, skipping the first 6 rows
file_path = "/mnt/data/pge_electric_usage_interval_data_Service 1_1_2025-04-23_to_2025-05-22.csv"
df = pd.read_csv(file_path, skiprows=6)

# Preview the dataframe
print(df.head())

# Convert Start Date and End Date columns to datetime
if 'Start Date' in df.columns:
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
else:
    raise ValueError("Expected 'Start Date' and 'End Date' columns not found.")

# Filter for data within this month (e.g., May 2025)
start_of_month = pd.Timestamp("2025-05-01")
end_of_month = pd.Timestamp("2025-05-31")
df_month = df[(df['Start Date'] >= start_of_month) & (df['Start Date'] < end_of_month)]

# Plot the kWh usage over time
plt.figure(figsize=(12, 6))
plt.plot(df_month['Start Date'], df_month['Usage (kWh)'], marker='o', linestyle='-')
plt.title("Electric Usage (kWh) for May 2025")
plt.xlabel("Date")
plt.ylabel("Usage (kWh)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
