from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
BATTERY_CAPACITY_KWH = 13.5  # usable capacity
BATTERY_MAX_CHARGE_KW = 5.0  # max charge rate (kWh per hour)
FILTER_START = pd.Timestamp("2025-01-01")

# EV charging constants
EV_WEEKLY_TARGET_KWH = 46.0  # target weekly charging
EV_CHARGE_DAYS = (6, 2, 4, 5)  # Sun=6, Wed=2, Fri=4, Sat=5 (pandas dayofweek: Mon=0..Sun=6)
EV_MAX_CHARGE_KW = 11.5  # max EV charge rate


def load_usage(path: str | Path) -> pd.DataFrame:
    """Load demand data with component columns (base, water heat)."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['DATE'])
    df['base_kwh'] = pd.to_numeric(df['base_kwh'], errors='coerce')
    df['water_heat_kwh'] = pd.to_numeric(df['water_heat_kwh'], errors='coerce')
    df['usage_kwh'] = df['base_kwh'] + df['water_heat_kwh']  # total non-EV usage
    return df[['timestamp', 'base_kwh', 'water_heat_kwh', 'usage_kwh']]

def load_generation(path: str | Path) -> pd.DataFrame:
    """Load PV generation data from the PVWatts output file."""

    # Read from that header row (after removing PII rows from PVWatts export)
    df = pd.read_csv(path, skiprows=26)
    # Build timestamp for each row (year = 2025)
    df['timestamp'] = pd.to_datetime({
        'year': FILTER_START.year,
        'month': df['Month'],
        'day': df['Day'],
        'hour': df['Hour'],
    })
    # Convert AC System Output (W) to kWh per hour
    df['generation_kwh'] = df['AC System Output (W)'] / 1000.0
    return df[['timestamp', 'generation_kwh']]


MAX_DEMAND_KW = 11.5  # Maximum allowed hourly demand


def simulate_battery(
    usage_fp: str | Path,
    generation_fp: str | Path,
    capacity_kwh: float = BATTERY_CAPACITY_KWH,
) -> pd.DataFrame:
    """Merge usage and generation, then simulate battery SOC starting fully charged."""
    df_use = load_usage(usage_fp)
    df_gen = load_generation(generation_fp)
    # Align on timestamp
    df = pd.merge(df_use, df_gen, on='timestamp', how='inner')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Check max demand
    max_usage = df['usage_kwh'].max()
    if max_usage > MAX_DEMAND_KW:
        print(f"⚠️  Warning: Max hourly demand ({max_usage:.2f} kWh) exceeds {MAX_DEMAND_KW} kW limit")
        over_limit = df[df['usage_kwh'] > MAX_DEMAND_KW][['timestamp', 'usage_kwh']]
        print(f"   {len(over_limit)} hours exceed limit. First 5:")
        print(over_limit.head().to_string(index=False))
    else:
        print(f"✓ Max hourly demand: {max_usage:.2f} kWh (under {MAX_DEMAND_KW} kW limit)")

    # Start battery fully charged
    soc = capacity_kwh
    soc_list = []
    battery_charge_list = []
    battery_discharge_list = []
    ev_charge_list = []
    grid_export_list = []
    grid_import_list = []

    # EV charging tracking
    current_week = None
    ev_charged_this_week = 0.0
    weekly_ev_summary = []  # list of (week, charged_at_home, needed_at_work)

    for _, row in df.iterrows():
        gen = row['generation_kwh']
        usage = row['usage_kwh']
        ts = row['timestamp']
        day_of_week = ts.dayofweek
        week = ts.to_period('W-SUN')

        # Track weekly EV charging - reset at start of new week
        if current_week is not None and week != current_week:
            # Save previous week's summary
            needed_at_work = max(0, EV_WEEKLY_TARGET_KWH - ev_charged_this_week)
            weekly_ev_summary.append((current_week, ev_charged_this_week, needed_at_work))
            ev_charged_this_week = 0.0
        current_week = week

        # Step 1: Solar covers usage first
        gen_to_usage = min(gen, usage)
        remaining_gen = gen - gen_to_usage
        remaining_usage = usage - gen_to_usage

        # Step 2: Remaining solar charges battery (capped by rate and capacity)
        battery_charge = min(remaining_gen, capacity_kwh - soc, BATTERY_MAX_CHARGE_KW)
        soc += battery_charge
        remaining_gen -= battery_charge

        # Step 3: EV charging from surplus (only on eligible days, up to weekly target and charge rate)
        ev_charge = 0.0
        if day_of_week in EV_CHARGE_DAYS and remaining_gen > 0:
            ev_available = EV_WEEKLY_TARGET_KWH - ev_charged_this_week
            ev_charge = min(remaining_gen, ev_available, EV_MAX_CHARGE_KW)
            ev_charged_this_week += ev_charge
            remaining_gen -= ev_charge

        # Step 4: Remaining solar is exported
        grid_export = remaining_gen

        # Step 5: Remaining usage is covered by battery
        battery_discharge = min(remaining_usage, soc)
        soc -= battery_discharge
        remaining_usage -= battery_discharge

        # Step 6: Any remaining usage comes from grid
        grid_import = remaining_usage

        soc_list.append(soc)
        battery_charge_list.append(battery_charge)
        battery_discharge_list.append(battery_discharge)
        ev_charge_list.append(ev_charge)
        grid_export_list.append(grid_export)
        grid_import_list.append(grid_import)

    # Save final week's summary
    if current_week is not None:
        needed_at_work = max(0, EV_WEEKLY_TARGET_KWH - ev_charged_this_week)
        weekly_ev_summary.append((current_week, ev_charged_this_week, needed_at_work))

    df['battery_soc_kwh'] = soc_list
    df['battery_charge_kwh'] = battery_charge_list
    df['battery_discharge_kwh'] = battery_discharge_list
    df['ev_charge_kwh'] = ev_charge_list
    df['grid_export_kwh'] = grid_export_list
    df['grid_import_kwh'] = grid_import_list

    # Total demand including EV
    df['total_demand_kwh'] = df['base_kwh'] + df['water_heat_kwh'] + df['ev_charge_kwh']

    # Build EV summary dataframe
    ev_summary_df = pd.DataFrame(weekly_ev_summary, columns=['week', 'charged_at_home', 'needed_at_work'])
    total_home = ev_summary_df['charged_at_home'].sum()
    total_work = ev_summary_df['needed_at_work'].sum()
    total_target = len(ev_summary_df) * EV_WEEKLY_TARGET_KWH

    # EV charge rate sanity check
    max_ev_charge = df['ev_charge_kwh'].max()
    if max_ev_charge > EV_MAX_CHARGE_KW:
        print(f"⚠️  Warning: Max EV charge rate ({max_ev_charge:.2f} kWh) exceeds {EV_MAX_CHARGE_KW} kW limit")
    else:
        print(f"✓ Max EV charge rate: {max_ev_charge:.2f} kWh (under {EV_MAX_CHARGE_KW} kW limit)")

    print(f"\n🚗 EV Charging Summary:")
    print(f"   Weeks simulated:       {len(ev_summary_df)}")
    print(f"   Weekly target:         {EV_WEEKLY_TARGET_KWH} kWh")
    print(f"   Total charged at home: {total_home:,.1f} kWh ({100*total_home/total_target:.1f}% of target)")
    print(f"   Total needed at work:  {total_work:,.1f} kWh ({100*total_work/total_target:.1f}% of target)")

    # Store summary on the dataframe for access
    df.attrs['ev_summary'] = ev_summary_df
    # Filter to 2025 onward
    df = df[df['timestamp'] >= FILTER_START].reset_index(drop=True)
    return df


def plot_energy(df: pd.DataFrame, index: int, short_desc: str):
    """Plot usage, generation, and battery SOC."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['usage_kwh'], label='Usage (kWh)')
    plt.plot(df['timestamp'], df['generation_kwh'], label='Generation (kWh)')
    plt.plot(df['timestamp'], df['battery_soc_kwh'], label='Battery SOC (kWh)')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.title(f'Battery Simulation {short_desc}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()


if __name__ == '__main__':
    # Paths for actual install
    usage_file = Path('../output/260204_2025_adjusted_kWh.csv')
    gen_file = Path('../input/install/250204_pvwatts_hourly.csv')
    output_dir = Path('../output/install/')
    figs_dir = Path('../figs/install/')

    # Create output directories if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation
    df = simulate_battery(usage_file, gen_file)

    # Save results - detailed version with demand breakdown
    detail_cols = [
        'timestamp',
        # Demand sources
        'base_kwh', 'water_heat_kwh', 'ev_charge_kwh', 'total_demand_kwh',
        # Generation
        'generation_kwh',
        # Battery state
        'battery_soc_kwh', 'battery_charge_kwh', 'battery_discharge_kwh',
        # Grid interaction
        'grid_import_kwh', 'grid_export_kwh',
    ]
    out_file = output_dir / '260204_merged_battery.csv'
    df[detail_cols].to_csv(out_file, index=False)
    print(f"Saved battery simulation to {out_file}")

    # Save EV weekly summary
    ev_summary = df.attrs.get('ev_summary')
    if ev_summary is not None:
        ev_file = output_dir / '260204_ev_weekly_summary.csv'
        ev_summary.to_csv(ev_file, index=False)
        print(f"Saved EV weekly summary to {ev_file}")

    # Plot and save figure
    plot_energy(df, 0, 'install')
    fig_path = figs_dir / '260204_battery_simulation.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved plot to {fig_path}")
