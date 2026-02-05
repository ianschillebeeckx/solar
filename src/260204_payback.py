import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants for actual install
SYSTEM_SIZE_KW = 6.6
TOTAL_CAPITAL = 26848  # total installed cost (solar + battery)
ANNUAL_INCREASE = 0.06  # electricity price increase rate
DEGRADATION_RATE = 0.005  # solar panel degradation rate (0.5%/year)
YEARS = 20

# Paths
base_dir = Path('../output/install/')
figs_dir = Path('../figs/install/')
figs_dir.mkdir(parents=True, exist_ok=True)

# Load electricity costs for install scenario
elec_cost_df = pd.read_csv(base_dir / '260204_electricity_costs.csv')

# Compute baseline cost (what you'd pay without solar)
BASELINE_COST = (elec_cost_df["total_demand_kwh"] * elec_cost_df["buy_price"]).sum()

# Load annual cost with solar from summary
summary_df = pd.read_csv(base_dir / '260204_summary.csv')
ANNUAL_COST_WITH_SOLAR = summary_df['annual_cost'].iloc[0]


def cumulative_savings_degraded(initial_savings, growth_rate, degradation_rate, years):
    """Compute cumulative savings with compound growth and panel degradation."""
    q = (1 + growth_rate) * (1 - degradation_rate)
    if np.isclose(q, 1.0):
        return initial_savings * years
    return initial_savings * (1 - q**years) / (1 - q)


def compute_payback(initial_savings, growth_rate, degradation_rate, capital_cost, max_years=30):
    """Compute payback period with interpolation."""
    q = (1 + growth_rate) * (1 - degradation_rate)
    cumulative = 0.0
    for n in range(1, max_years + 1):
        s_n = initial_savings * (q ** (n - 1))
        prev = cumulative
        cumulative += s_n
        if cumulative >= capital_cost:
            fraction = (capital_cost - prev) / s_n
            return round((n - 1) + fraction, 1)
    return np.nan


# Calculate metrics
initial_savings = BASELINE_COST - ANNUAL_COST_WITH_SOLAR
cumulative_savings = cumulative_savings_degraded(initial_savings, ANNUAL_INCREASE, DEGRADATION_RATE, YEARS)
profit = cumulative_savings - TOTAL_CAPITAL
payback_years = compute_payback(initial_savings, ANNUAL_INCREASE, DEGRADATION_RATE, TOTAL_CAPITAL)

# Print summary
print(f"=== Payback Analysis for {SYSTEM_SIZE_KW} kW Install ===")
print(f"Baseline annual cost (no solar): ${BASELINE_COST:,.2f}")
print(f"Annual cost with solar: ${ANNUAL_COST_WITH_SOLAR:,.2f}")
print(f"Initial annual savings: ${initial_savings:,.2f}")
print(f"Total capital cost: ${TOTAL_CAPITAL:,.2f}")
print(f"Payback period: {payback_years} years")
print(f"Cumulative savings over {YEARS} years: ${cumulative_savings:,.2f}")
print(f"Profit after {YEARS} years: ${profit:,.2f}")

# Plot cumulative savings vs capital cost over time
years_range = np.arange(0, YEARS + 1)
q = (1 + ANNUAL_INCREASE) * (1 - DEGRADATION_RATE)
cumulative_by_year = [0]
for y in range(1, YEARS + 1):
    cumulative_by_year.append(cumulative_savings_degraded(initial_savings, ANNUAL_INCREASE, DEGRADATION_RATE, y))

plt.figure(figsize=(10, 6))
plt.plot(years_range, cumulative_by_year, marker='o', label='Cumulative Savings')
plt.axhline(y=TOTAL_CAPITAL, color='r', linestyle='--', label=f'Capital Cost (${TOTAL_CAPITAL:,.0f})')
if not np.isnan(payback_years):
    plt.axvline(x=payback_years, color='g', linestyle=':', label=f'Payback ({payback_years} yrs)')
plt.xlabel('Years')
plt.ylabel('Dollars ($)')
plt.title(f'Solar Payback Analysis - {SYSTEM_SIZE_KW} kW System')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig_path = figs_dir / '260204_payback.png'
plt.savefig(fig_path)
print(f"Saved payback plot to {fig_path}")
plt.show()
