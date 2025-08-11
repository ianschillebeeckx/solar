import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from zoneinfo import ZoneInfo
from astral.sun import sun
from astral import Observer

"""
Weekly **average daily energy** usage — *from 2025‑01‑01 onward*

Lines plotted
* **Day – Electric‑only** (dashed, royal‑blue)
* **Night – Electric‑only** (dashed, slate‑blue)
* **Day – Total (Electric + Gas‑equiv)** (solid, firebrick)
* **Night – Total (Electric + Gas‑equiv)** (solid, dark‑orange)

**Gas distribution logic**  
Gas readings are *daily totals* (therms). Steps:
1. Convert therms → kWh with 24.095 kWh / therm.
2. Split evenly across 24 hours of that day.
3. Allocate to *Day* vs *Night* using daylight‑hour fraction.

Electric interval readings keep their actual timestamps. Secondary Y‑axis shows
average daylight hours. X‑axis ticks mark first week‑ending of each month.
"""

# ── Configuration ──────────────────────────────────────────────────────────
CSV_PATH   = Path('../input/pge_usage_2024-08-01_to_2025-07-31.csv')
MIN_DATE   = '2025-01-01'
TZ         = ZoneInfo('America/Los_Angeles')
OBSERVER   = Observer(latitude=37.7749, longitude=-122.4194, elevation=0)
THERM_TO_KWH = 29.3 * (0.4 * 0.60 / 0.85 + 0.6 * 0.90)   # ≈ 24.095

# ── Sun helpers ───────────────────────────────────────────────────────────
_sun_cache: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}

def sunrise_sunset(date_ts: pd.Timestamp):
    d = date_ts.normalize()
    if d not in _sun_cache:
        t = sun(OBSERVER, date=d.date(), tzinfo=TZ)
        _sun_cache[d] = (t['sunrise'], t['sunset'])
    return _sun_cache[d]

def daylight_hours(date_ts: pd.Timestamp) -> float:
    sr, ss = sunrise_sunset(date_ts)
    return (ss - sr).total_seconds() / 3600.0

def classify_period(ts: pd.Timestamp):
    sr, ss = sunrise_sunset(ts)
    return 'Day' if sr <= ts < ss else 'Night'

# ── Load data ─────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_PATH)
naive = pd.to_datetime(df_raw['DATE'] + ' ' + df_raw['START TIME'], format='%Y-%m-%d %H:%M')
df_raw['DateTime'] = naive.dt.tz_localize(TZ, ambiguous='NaT', nonexistent='shift_forward')
df_raw = df_raw[df_raw['DateTime'].notna()]
df_raw = df_raw[df_raw['DateTime'] >= pd.Timestamp(MIN_DATE, tz=TZ)]

usage_col = next((c for c in df_raw.columns if 'USAGE' in c.upper()), None)
if not usage_col:
    raise KeyError('Usage column not found')

df_raw[usage_col] = pd.to_numeric(df_raw[usage_col], errors='coerce')
_df_type = df_raw['TYPE'].str.lower()

# ── Electric subset ───────────────────────────────────────────────────────
df_elec = df_raw[_df_type.str.contains('electric')].copy()
df_elec['Period'] = df_elec['DateTime'].apply(classify_period)
df_elec['kWh']    = df_elec[usage_col]  # already kWh

daily_elec = (
    df_elec.groupby([pd.Grouper(key='DateTime', freq='D'), 'Period'])['kWh']
            .sum()
            .unstack(fill_value=0)
)

# ── Gas subset → daily Day/Night split ────────────────────────────────────
df_gas = df_raw[_df_type.str.contains('gas')].copy()
df_gas['Date'] = df_gas['DateTime'].dt.normalize()

df_gas_daily = df_gas.groupby('Date')[usage_col].sum().to_frame('therms')
df_gas_daily['kWh_total'] = df_gas_daily['therms'] * THERM_TO_KWH

# Ensure index is tz‑aware in TZ
if df_gas_daily.index.tz is None:
    _idx = df_gas_daily.index.tz_localize(TZ)
else:
    _idx = df_gas_daily.index.tz_convert(TZ)

df_gas_daily.index = _idx

df_gas_daily['daylight_hours'] = [daylight_hours(d) for d in df_gas_daily.index]
df_gas_daily['day_frac'] = df_gas_daily['daylight_hours'] / 24.0

df_gas_daily['Day']   = df_gas_daily['kWh_total'] * df_gas_daily['day_frac']
df_gas_daily['Night'] = df_gas_daily['kWh_total'] * (1 - df_gas_daily['day_frac'])

daily_gas = df_gas_daily[['Day', 'Night']]

# ── Combine & resample ────────────────────────────────────────────────────
daily_total = daily_elec.add(daily_gas, fill_value=0)
weekly_electric = daily_elec.resample('W-SUN').mean()
weekly_total   = daily_total.resample('W-SUN').mean()

# ── Weekly daylight hours for secondary axis ──────────────────────────────
all_days = pd.date_range(weekly_total.index.min().normalize(), weekly_total.index.max().normalize(), tz=TZ, freq='D')
weekly_daylight = pd.Series([daylight_hours(d) for d in all_days], index=all_days).resample('W-SUN').mean().reindex(weekly_total.index)

# ── X‑ticks per month ─────────────────────────────────────────────────────
month_change = weekly_total.index.to_period('M').to_series().diff().fillna(1) != 0
month_ticks  = weekly_total.index[month_change]
month_labels = [d.strftime('%b %Y') for d in month_ticks]

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(weekly_electric.index, weekly_electric.get('Day', 0),  label='Day Electric kWh/d',  linestyle='--', linewidth=1.8, color='#1f77b4')
ax1.plot(weekly_electric.index, weekly_electric.get('Night', 0), label='Night Electric kWh/d', linestyle='--', linewidth=1.8, color='#5b8ffb')

ax1.plot(weekly_total.index,   weekly_total.get('Day', 0),    label='Day Total kWh/d',  linewidth=2.4, color='#d62728')
ax1.plot(weekly_total.index,   weekly_total.get('Night', 0),  label='Night Total kWh/d', linewidth=2.4, color='#ff8c2f')

ax1.set_ylabel('Average Daily Energy (kWh)')
ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.set_xticks(month_ticks)
ax1.set_xticklabels(month_labels, rotation=45, ha='right')
ax1.set_xlabel('Month (tick = first week‑ending)')

ax2 = ax1.twinx()
ax2.plot(weekly_daylight.index, weekly_daylight, label='Avg Sunlight Hours', color='grey', linestyle=':', marker='o', alpha=0.6)
ax2.set_ylabel('Avg Sunlight Hours / Day')

fig.suptitle('Weekly Avg Daily Energy (Electric vs Total) & Daylight — 2025 (SF)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
