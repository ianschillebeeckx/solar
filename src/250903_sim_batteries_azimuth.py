import pandas as pd
import calendar
import matplotlib.pyplot as plt

# Constants
BATTERY_MAX_SOC = 13.5  # Full capacity threshold for selling


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
    buy.index = sell.index.astype(int)

    return batt, sell, buy


def process_and_calculate(batt: pd.DataFrame, sell: pd.DataFrame, buy: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate batt, compute buys/sells and net cost per hour.
    """
    df = batt.copy()
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['prev_soc'] = df['battery_soc_kwh'].shift(1).fillna(df['battery_soc_kwh'])

    def lookup_price(hour, month, lookup):
        mon_abbr = calendar.month_abbr[month]
        return lookup.at[int(hour), mon_abbr]

    def cost_row(row):
        soc = row['battery_soc_kwh']
        prev_soc = row['prev_soc']
        usage = row['usage_kwh']
        gen = row['generation_kwh']
        sell_price = lookup_price(row['hour'], row['month'], sell)
        buy_kwh = 0.0
        sell_kwh = 0.0
        buy_price = lookup_price(row['hour'], row['month'], buy)
        cost = 0.0
        # Purchase logic
        if soc == 0 and (usage - gen - prev_soc) > 0:
            buy_kwh = usage - gen - prev_soc
            cost = buy_kwh * buy_price
        # No trade if battery has charge and there is usage
        elif soc >0 and soc < BATTERY_MAX_SOC and usage > 0:
            pass
        # Selling logic
        elif soc >= BATTERY_MAX_SOC and (gen - usage) > 0:
            sell_kwh = gen - usage
            cost = -sell_kwh * sell_price
        # night time, no usage
        elif gen == 0 and usage == 0:
            buy_kwh = 0
            cost = 0
        else:
            print("something else happened")
            print(row[['timestamp','usage_kwh','generation_kwh','battery_soc_kwh']])

        return pd.Series({
            'buy_kwh': buy_kwh,
            'buy_price': buy_price,
            'sell_kwh': sell_kwh,
            'sell_price': sell_price,
            'cost': cost
        })

    trades = df.apply(cost_row, axis=1)
    return pd.concat([df, trades], axis=1)


def plot_metrics(df: pd.DataFrame, kw: float, scenario: int, azi: int):
    """
    Plot generation, usage, battery SOC, and net cost over time for a scenario.
    """
    plt.figure(figsize=(10,6))
    plt.plot(df['timestamp'], df['generation_kwh'], label='Generation (kWh)')
    plt.plot(df['timestamp'], df['usage_kwh'], label='Usage (kWh)')
    plt.plot(df['timestamp'], df['battery_soc_kwh'], label='Battery SOC (kWh)')
    plt.plot(df['timestamp'], df['cost'], label='Net Cost ($)')
    plt.legend()
    plt.ylim([-2,14])
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title(f'Scenario {kw}-{scenario}-{azi}: Metrics Over Time')
    plt.tight_layout()


if __name__ == '__main__':
    sell_price_file = '../input/250810 PGE Comp.csv'
    buy_price_file = '../input/250814 PGE ELEC.csv'


    kws = [4.5, 5.5, 6.4, 7.5]
    scenarios = [20, 40]
    kw = 6.4
    azis = [120,140,160,180,200,220,240,260]

    costs = pd.DataFrame(columns=azis, index=scenarios)
    for s in scenarios:

        base_dir = f'../output/kw_{kw}/'
        fig_path = f'../figs/kw_{kw}/250903_lineplot_monthly_cost_kw_{kw}_{s}_azi.png'
        monthly_costs = {}

        for azi in azis:
            batt_file = f'{base_dir}merged_battery_2025_onward_{s}_{azi}.csv'
            output_file = f'{base_dir}electricity_costs_{s}_{azi}.csv'

            # Load and process
            batt_df, sell_df, buy_df = load_data(batt_file, sell_price_file, buy_price_file)
            result_df = process_and_calculate(batt_df, sell_df, buy_df)

            # Save results
            cols = [
                'timestamp', 'usage_kwh', 'generation_kwh', 'battery_soc_kwh',
                'buy_kwh', 'buy_price', 'sell_kwh', 'sell_price', 'cost'
            ]
            result_df.to_csv(output_file, columns=cols, index=False)

            # Plot metrics for each scenario
            plot_metrics(result_df, kw, s, azi)
            plt.savefig(f'../figs/kw_{kw}/250903_lineplot_pv_{s}_{azi}.png')

            # Aggregate monthly cost and print total
            monthly = result_df.groupby(result_df['timestamp'].dt.to_period('M'))['cost'].sum()
            monthly.index = monthly.index.to_timestamp()
            monthly_costs[azi] = monthly
            total_cost = monthly.sum()
            costs.at[s,azi] = total_cost
            print(f"=>Scenario {s} {azi}: total net cost across all months = ${total_cost:,.2f}")

        # Plot monthly costs comparison
        plt.figure(figsize=(10,6))
        for a, m in monthly_costs.items():
            plt.plot(m.index, m.values, marker='o', linestyle='-', label=f'Azimuth {a}')
        # plt.ylim([0,275])
        plt.xlabel('Month')
        plt.ylabel('Total Net Cost ($)')
        plt.title(f'Monthly Net Cost Comparison {kw} {s}')
        plt.legend()
        plt.tight_layout()
        # Save figure
        plt.savefig(fig_path)
        print(f"Monthly cost comparison figure saved to {fig_path}")

    costs.to_csv('../output/250903_all_costs_azi.csv')
    plt.show()