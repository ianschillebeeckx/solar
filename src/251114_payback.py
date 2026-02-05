from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGS_DIR = PROJECT_ROOT / "figs"

ALL_COSTS_CSV = OUTPUT_DIR / "251107_all_costs.csv"
BASELINE_COSTS_CSV = OUTPUT_DIR / "kw_6.4/251107_electricity_costs_0.csv"
FIG_PATH = FIGS_DIR / "251114_payback.png"

# Capital assumptions (same as prior version)
COST_PER_KW = (500 + 25456 - 5772.80) / 7.04  # ≈2866
BATTERY_COST = 500 + 14950 - 2000 - 5940      # ≈7510
CAPITAL_OVERRIDES = {
    5.5: 24549,
    6.4: 26338,
    6.9: 27693,
}
ANNUAL_INCREASE = 0.06
DEGRADATION_RATE = 0.005
YEARS = 20


def load_cost_tables() -> tuple[pd.DataFrame, float]:
    """Return melted `all_costs` frame and baseline annual utility bill."""
    elec_cost_df = pd.read_csv(BASELINE_COSTS_CSV)
    baseline_cost = (elec_cost_df["usage_kwh"] * elec_cost_df["buy_price"]).sum()

    df = pd.read_csv(ALL_COSTS_CSV).rename(columns={"Unnamed: 0": "Tilt"})
    melted = df.melt(id_vars=["Tilt"], var_name="Size_kW", value_name="Annual_Cost")
    melted["Size_kW"] = melted["Size_kW"].astype(float)
    melted["Capital_Cost"] = melted["Size_kW"].map(CAPITAL_OVERRIDES).fillna(
        melted["Size_kW"] * COST_PER_KW + BATTERY_COST
    )
    melted["Initial_Savings"] = baseline_cost - melted["Annual_Cost"]
    return df, melted


def cumulative_savings_degraded(initial_savings: float) -> float:
    """Closed-form geometric sum of savings with growth & degradation."""
    q = (1 + ANNUAL_INCREASE) * (1 - DEGRADATION_RATE)
    if np.isclose(q, 1.0):
        return initial_savings * YEARS
    return initial_savings * (1 - q ** YEARS) / (1 - q)


def compute_payback(initial_savings: float, capital_cost: float, max_years: int = 30) -> float:
    """Return payback period in years (tenths) based on degraded savings."""
    q = (1 + ANNUAL_INCREASE) * (1 - DEGRADATION_RATE)
    cumulative = 0.0
    for n in range(1, max_years + 1):
        savings_year_n = initial_savings * (q ** (n - 1))
        prev = cumulative
        cumulative += savings_year_n
        if cumulative >= capital_cost:
            frac = (capital_cost - prev) / savings_year_n
            return round((n - 1) + frac, 1)
    return np.nan


def build_summary_tables(df_raw: pd.DataFrame, melted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    melted["Cumulative_Savings_20yr"] = melted["Initial_Savings"].apply(cumulative_savings_degraded)
    melted["Profit_20yr"] = melted["Cumulative_Savings_20yr"] - melted["Capital_Cost"]
    melted["Payback_Years"] = melted.apply(
        lambda row: compute_payback(row["Initial_Savings"], row["Capital_Cost"]),
        axis=1,
    )

    profit_pivot_raw = melted.pivot(index="Tilt", columns="Size_kW", values="Profit_20yr")
    profit_display = profit_pivot_raw.map(lambda x: f"${x:,.0f}")
    payback_pivot = melted.pivot(index="Tilt", columns="Size_kW", values="Payback_Years")
    annual_cost_pivot = df_raw.set_index("Tilt")

    return annual_cost_pivot, payback_pivot, profit_pivot_raw, profit_display


def plot_heatmaps(annual_cost: pd.DataFrame, payback: pd.DataFrame, profit: pd.DataFrame, profit_fmt: pd.DataFrame) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    sns.heatmap(annual_cost, ax=axes[0], annot=True, fmt=".0f", cmap="Greys", cbar=False)
    axes[0].set_title("Annual Electricity Cost")
    axes[0].set_xlabel("Solar Size (kW)")
    axes[0].set_ylabel("Tilt (Degrees)")

    sns.heatmap(payback, ax=axes[1], annot=True, fmt=".1f", cmap="YlOrRd", cbar=False)
    axes[1].set_title("Payback Period (Years)")
    axes[1].set_xlabel("Solar Size (kW)")
    axes[1].set_ylabel("")
    axes[1].set_yticks([])

    sns.heatmap(profit, ax=axes[2], annot=profit_fmt, fmt="", cmap="YlGnBu", cbar=False)
    axes[2].set_title("Profit After 20 Years")
    axes[2].set_xlabel("Solar Size (kW)")
    axes[2].set_ylabel("")
    axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved heatmaps to {FIG_PATH}")


def main() -> None:
    df_raw, melted = load_cost_tables()
    annual_cost, payback, profit_raw, profit_fmt = build_summary_tables(df_raw, melted)
    plot_heatmaps(annual_cost, payback, profit_raw, profit_fmt)


if __name__ == "__main__":
    main()
