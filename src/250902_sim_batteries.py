import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
BATTERY_CAPACITY_KWH = 13.5  # usable capacity
FILTER_START = pd.Timestamp("2025-01-01")


def load_usage(path: str | Path) -> pd.DataFrame:
    """Load demand data from the adjusted_kWh file."""
    # Tab-delimited file with DATE including timestamp and adjusted_kwh
    df = pd.read_csv(path)
    # Parse the DATE column as timestamp directly
    df['timestamp'] = pd.to_datetime(df['DATE'])
    # Use adjusted_kwh for usage (numeric)
    df['usage_kwh'] = pd.to_numeric(df['adjusted_kwh'], errors='coerce')
    return df[['timestamp', 'usage_kwh']]

def load_generation(path: str | Path) -> pd.DataFrame:
    """Load PV generation data from the PVWatts output file."""

    # Read from that header row
    df = pd.read_csv(path, skiprows=31)
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

    # Start battery fully charged
    soc = capacity_kwh
    soc_list = []
    for _, row in df.iterrows():
        diff = row['generation_kwh'] - row['usage_kwh']
        if diff >= 0:
            charge = min(diff, capacity_kwh - soc)
            soc += charge
        else:
            discharge = min(-diff, soc)
            soc -= discharge
        soc_list.append(soc)

    df['battery_soc_kwh'] = soc_list
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
    filter_start = pd.Timestamp("2025-01-01")

    for kw in [5.5, 6.4, 6.9, 7.5]:
        # Local paths and settings (lowercase names)
        generation_dir = Path(f'../input/kw_{kw}')
        output_dir = Path(f'../output/kw_{kw}/')
        figs_dir = Path(f'../figs/kw_{kw}/')
        generations = [0, 10, 20, 30, 40, 50, 60, 70]

        # File definitions
        usage_file = '../output/250902_2025_adjusted_KWh.csv'

        # Batch simulate and save
        for i in generations:
            gen_file = generation_dir / f'pvwatts_hourly_{i}.csv'
            out_file = output_dir / f'merged_battery_2025_onward_{i}.csv'
            df = simulate_battery(usage_file, gen_file)
            df.to_csv(out_file, index=False)
            short_desc = f'pv{i}'

            # Plot and save each figure
            plot_energy(df, i, short_desc)
            fig_path = figs_dir / f'250902_lineplot_{short_desc}.png'
            plt.savefig(fig_path)
            plt.close()
            print(f"Saved plot to {fig_path}")