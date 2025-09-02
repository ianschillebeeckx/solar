import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../output/2025_adjusted_kWh.csv')
df["DATE"] = pd.to_datetime(df["DATE"])

# Group by month, sum adjusted_kwh and original_kwh
monthly_both = (
    df.groupby(df["DATE"].dt.to_period("M"))[["adjusted_kwh","original_kwh"]]
      .sum()
      .reset_index()
)
monthly_both["DATE"] = monthly_both["DATE"].dt.to_timestamp()

# Plot grouped bar chart
x = range(len(monthly_both))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar([i - width/2 for i in x], monthly_both["original_kwh"], width=width, label="Original kWh")
plt.bar([i + width/2 for i in x], monthly_both["adjusted_kwh"], width=width, label="Adjusted kWh")
plt.xticks(x, monthly_both["DATE"].dt.strftime("%Y-%m"), rotation=45, ha="right")
plt.ylabel("Total kWh")
plt.title("Monthly Total: Original vs Adjusted kWh")
plt.legend()
plt.tight_layout()
plt.show()
