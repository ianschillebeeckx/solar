import pandas as pd
import calendar
import matplotlib.pyplot as plt

# Constants
MONTHLY_FEE = 15.0  # flat monthly fee in dollars


def load_data(battery_path: str, sell_price_path: str, buy_price_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Load true CSV battery data (comma-separated) and sell-price schedule.
    """
    batt = pd.read_csv(
        battery_path,
        parse_dates=['timestamp'],
        dayfirst=False
    )
    sell = pd.read_csv(sell_price_path, index_col=0)
    sell.index = sell.index.astype(int)

    buy = pd.read_csv(buy_price_path, index_col=0)
    # Optional tiny fix: make sure we cast the BUY index, not SELL's
    buy.index = buy.index.astype(int)

    return batt, sell, buy


def process_and_calculate(batt: pd.DataFrame, sell: pd.DataFrame, buy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate costs using grid_import_kwh and grid_export_kwh from battery simulation.
    """
    df = batt.copy()
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour

    def lookup_price(hour, month, lookup):
        mon_abbr = calendar.month_abbr[month]
        return lookup.at[int(hour), mon_abbr]

    # Look up prices for each row
    df['buy_price'] = df.apply(lambda r: lookup_price(r['hour'], r['month'], buy), axis=1)
    df['sell_price'] = df.apply(lambda r: lookup_price(r['hour'], r['month'], sell), axis=1)

    # Use grid_import/export directly from battery simulation
    df['buy_kwh'] = df['grid_import_kwh']
    df['sell_kwh'] = df['grid_export_kwh']

    # Calculate cost: positive = paid to utility, negative = credit from utility
    df['cost'] = (df['buy_kwh'] * df['buy_price']) - (df['sell_kwh'] * df['sell_price'])

    return df


def append_monthly_fee_rows(df: pd.DataFrame, monthly_fee: float = MONTHLY_FEE) -> pd.DataFrame:
    """
    Append one synthetic row per month with a flat fee in 'cost'.
    Timestamp rule: last second of each month (e.g., 2025-01-31 23:59:59).
    This keeps the fee inside the correct month for groupby-to-period('M') operations.
    """
    if df.empty:
        return df

    months = df['timestamp'].dt.to_period('M').unique()
    fee_rows = []
    for p in months:
        # last second of the month
        ts = (p + 1).to_timestamp() - pd.Timedelta(seconds=1)

        fee_rows.append({
            'timestamp': ts,
            'base_kwh': 0.0,
            'water_heat_kwh': 0.0,
            'ev_charge_kwh': 0.0,
            'total_demand_kwh': 0.0,
            'generation_kwh': 0.0,
            'battery_soc_kwh': float(
                df.loc[df['timestamp'].dt.to_period('M') == p, 'battery_soc_kwh'].iloc[-1]
            ) if not df.loc[df['timestamp'].dt.to_period('M') == p].empty else 0.0,
            'buy_kwh': 0.0,
            'buy_price': 0.0,
            'sell_kwh': 0.0,
            'sell_price': 0.0,
            'cost': monthly_fee,
            'month': ts.month,
            'hour': ts.hour,
        })

    fee_df = pd.DataFrame(fee_rows)
    out = pd.concat([df, fee_df], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return out


def plot_metrics(df: pd.DataFrame, kw: float, scenario: int):
    """
    Plot generation, usage, battery SOC, and net cost over time for a scenario.
    """
    plt.figure(figsize=(10,6))
    plt.plot(df['timestamp'], df['generation_kwh'], label='Generation (kWh)')
    plt.plot(df['timestamp'], df['total_demand_kwh'], label='Total Demand (kWh)')
    plt.plot(df['timestamp'], df['battery_soc_kwh'], label='Battery SOC (kWh)')
    plt.plot(df['timestamp'], df['cost'], label='Net Cost ($)')
    plt.legend()
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title(f'Scenario {kw}-{scenario}: Metrics Over Time')
    plt.tight_layout()


if __name__ == '__main__':
    from pathlib import Path

    sell_price_file = '../input/250810 PGE Comp.csv'
    buy_price_file  = '../input/250905 PGE ELEC.csv'

    # Paths for actual install
    base_dir = Path('../output/install/')
    figs_dir = Path('../figs/install/')
    figs_dir.mkdir(parents=True, exist_ok=True)

    batt_file = base_dir / '260204_merged_battery.csv'
    output_file = base_dir / '260204_electricity_costs.csv'

    # Load and process
    batt_df, sell_df, buy_df = load_data(str(batt_file), sell_price_file, buy_price_file)
    result_df = process_and_calculate(batt_df, sell_df, buy_df)

    # Add $15/month fee rows (at month-end 23:59:59)
    result_df = append_monthly_fee_rows(result_df, monthly_fee=MONTHLY_FEE)

    # Save results
    cols = [
        'timestamp',
        'base_kwh', 'water_heat_kwh', 'ev_charge_kwh', 'total_demand_kwh',
        'generation_kwh', 'battery_soc_kwh',
        'buy_kwh', 'buy_price', 'sell_kwh', 'sell_price', 'cost'
    ]
    result_df.to_csv(output_file, columns=cols, index=False)
    print(f"Saved electricity costs to {output_file}")

    # Plot metrics
    plot_metrics(result_df, 6.6, 0)
    plt.savefig(figs_dir / '260204_metrics.png')
    plt.close()

    # Aggregate monthly cost and print total
    monthly = result_df.groupby(result_df['timestamp'].dt.to_period('M'))['cost'].sum()
    monthly.index = monthly.index.to_timestamp()
    total_cost = monthly.sum()
    print(f"Total net cost across all months = ${total_cost:,.2f}")

    # Plot monthly costs
    plt.figure(figsize=(10,6))
    plt.plot(monthly.index, monthly.values, marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Net Cost ($)')
    plt.title('Monthly Net Cost')
    plt.tight_layout()
    fig_path = figs_dir / '260204_monthly_cost.png'
    plt.savefig(fig_path)
    print(f"Monthly cost figure saved to {fig_path}")
    plt.close()

    # Save summary
    summary = pd.DataFrame({'annual_cost': [total_cost]})
    summary.to_csv(base_dir / '260204_summary.csv', index=False)
    # plt.show()
