import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Use seaborn's darkgrid style
sns.set(style="darkgrid")

# Constants
COST_PER_KW = 1729
BATTERY_COST = 10360
ANNUAL_INCREASE = 0.06
DEGRADATION_RATE = 0.005  # annual degradation rate of solar panel (0.5%)
YEARS = 15

# Compute baseline cost from external file
elec_cost_df = pd.read_csv("../output/kw_6.4/electricity_costs_0.csv")
BASELINE_COST = (elec_cost_df["usage_kwh"] * elec_cost_df["buy_price"]).sum()

# Load CSV data (update path as necessary)
df = pd.read_csv("../output/250905_all_costs.csv")
df = df.rename(columns={"Unnamed: 0": "Tilt"})

# Melt data for easier calculations
df_melted = df.melt(id_vars=["Tilt"], var_name="Size_kW", value_name="Annual_Cost")
df_melted["Size_kW"] = df_melted["Size_kW"].astype(float)

# Capital cost for each scenario
df_melted["Capital_Cost"] = df_melted["Size_kW"] * COST_PER_KW + BATTERY_COST

# Function to compute cumulative savings with compound growth and degradation (closed-form geometric sum)

def cumulative_savings_degraded(initial_savings, growth_rate, degradation_rate, years):
    # Savings each year follow S_t = S0 * q^t where q = (1+growth_rate)*(1-degradation_rate)
    q = (1 + growth_rate) * (1 - degradation_rate)
    if np.isclose(q, 1.0):
        return initial_savings * years
    return initial_savings * (1 - q**years) / (1 - q)

# Initial savings
df_melted["Initial_Savings"] = BASELINE_COST - df_melted["Annual_Cost"]

# Cumulative savings accounting for solar degradation
df_melted["Cumulative_Savings_15yr"] = df_melted["Initial_Savings"].apply(
    lambda x: cumulative_savings_degraded(x, ANNUAL_INCREASE, DEGRADATION_RATE, YEARS)
)

# Profit after 15 years
df_melted["Profit_15yr"] = df_melted["Cumulative_Savings_15yr"] - df_melted["Capital_Cost"]

# Format profit for display
profit_pivot_raw = df_melted.pivot(index="Tilt", columns="Size_kW", values="Profit_15yr")
profit_pivot = profit_pivot_raw.applymap(lambda x: f"${x:,.0f}")

# Function to compute payback period with tenths precision (discrete annual cash flows + within-year interpolation)
def compute_payback(initial_savings, growth_rate, degradation_rate, capital_cost, max_years=30):
    # Savings in year n follow a geometric series with ratio q = (1+growth_rate)*(1-degradation_rate)
    q = (1 + growth_rate) * (1 - degradation_rate)
    cumulative = 0.0
    for n in range(1, max_years + 1):
        s_n = initial_savings * (q ** (n - 1))  # savings during year n
        prev = cumulative
        cumulative += s_n
        if cumulative >= capital_cost:
            # Interpolate within year n assuming uniform accrual during the year
            fraction = (capital_cost - prev) / s_n
            return round((n - 1) + fraction, 1)
    return np.nan

df_melted["Payback_Years"] = df_melted.apply(
    lambda row: compute_payback(row["Initial_Savings"], ANNUAL_INCREASE, DEGRADATION_RATE, row["Capital_Cost"]),
    axis=1
)

# Pivot for heatmap plotting
payback_pivot = df_melted.pivot(index="Tilt", columns="Size_kW", values="Payback_Years")
annual_cost_pivot = df.set_index("Tilt")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Annual Cost heatmap
sns.heatmap(annual_cost_pivot, ax=axes[0], annot=True, fmt=".0f", cmap="Greys", cbar=False)
axes[0].set_title("Annual Electricity Cost")
axes[0].set_xlabel("Solar Size (kW)")
axes[0].set_ylabel("Tilt (Degrees)")

# Payback heatmap (now in the middle)
sns.heatmap(payback_pivot, ax=axes[1], annot=True, fmt=".1f", cmap="YlOrRd", cbar=False)
axes[1].set_title("Payback Period (Years)")
axes[1].set_xlabel("Solar Size (kW)")
axes[1].set_ylabel("")
axes[1].set_yticks([])

# Profit heatmap (now on the right)
sns.heatmap(profit_pivot_raw, ax=axes[2], annot=profit_pivot, fmt="", cmap="YlGnBu", cbar=False)
axes[2].set_title("Profit After 15 Years")
axes[2].set_xlabel("Solar Size (kW)")
axes[2].set_ylabel("")
axes[2].set_yticks([])

plt.tight_layout()
plt.show()
