import pandas as pd
from datetime import datetime

THERM_TO_KWH = 29.3 * (0.4 * 0.60 / 0.85 + 0.6 * 0.90)


def load_usage_data(file_path: str) -> pd.DataFrame:
    """Load the usage CSV into a DataFrame."""
    return pd.read_csv(file_path)


def process_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate electric and gas usage into daily totals."""
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Separate electric and gas
    electric_daily = (
        df[df['TYPE'].str.contains('Electric', case=False)]
        .groupby('DATE', as_index=False)['USAGE (kWh)']
        .sum()
        .rename(columns={'USAGE (kWh)': 'electric_kwh'})
    )

    gas_daily = (
        df[df['TYPE'].str.contains('Gas', case=False)]
        .groupby('DATE', as_index=False)['USAGE (kWh)']
        .sum()
        .rename(columns={'USAGE (kWh)': 'gas_therms'})
    )

    # Merge daily totals
    daily = pd.merge(electric_daily, gas_daily, on='DATE', how='outer').fillna(0)
    return daily


def convert_gas_to_kwh(gas_therms: float) -> float:
    """Convert gas usage in therms to kWh equivalent."""
    return gas_therms * THERM_TO_KWH


def adjust_hourly_usage(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Adjust hourly usage from start_date onward by adding converted gas to electric."""
    df['DATE'] = pd.to_datetime(df['DATE'])
    start_dt = pd.to_datetime(start_date)

    # Convert daily gas to kWh and then to hourly average
    df['converted_gas_kwh_hourly'] = df['gas_therms'].apply(convert_gas_to_kwh) / 24

    # Filter original hourly data and merge
    hourly_df = usage_df.copy()
    hourly_df['DATE'] = pd.to_datetime(hourly_df['DATE'])

    # Merge hourly with daily gas conversions
    hourly_df = hourly_df.merge(
        df[['DATE', 'converted_gas_kwh_hourly']],
        on='DATE',
        how='left'
    ).fillna(0)

    # Apply adjustment only from start_date onward
    hourly_df['adjusted_kwh'] = hourly_df['USAGE (kWh)']
    mask = hourly_df['DATE'] >= start_dt
    hourly_df.loc[mask, 'adjusted_kwh'] += hourly_df.loc[mask, 'converted_gas_kwh_hourly']

    return hourly_df


if __name__ == "__main__":
    file_path = '../input/pge_usage_2024-08-01_to_2025-07-31.csv'

    # Load data
    usage_df = load_usage_data(file_path)

    # Process daily totals
    daily_totals = process_daily_totals(usage_df)

    # Adjust hourly usage from Jan 1, 2025 onward
    adjusted_hourly = adjust_hourly_usage(daily_totals, start_date="2025-01-01")

    # Save to CSV
    output_path = '../output/2025_adjusted_kWh.csv'
    adjusted_hourly.to_csv(output_path, index=False)
    output_path
