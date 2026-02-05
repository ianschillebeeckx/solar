from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Target CSVs live under the repo's output/kw_* directories.
VARIANT_FILES = {
    "kw_5.5": Path("output/kw_5.5/251107_merged_battery_2025_onward_30.csv"),
    "kw_6.4": Path("output/kw_6.4/251107_merged_battery_2025_onward_30.csv"),
    "kw_6.9": Path("output/kw_6.9/251107_merged_battery_2025_onward_30.csv"),
    "kw_7.5": Path("output/kw_7.5/251107_merged_battery_2025_onward_30.csv"),
}

# Months to keep (calendar numbers) and thresholds (kWh) to test.
TARGET_MONTHS = (11, 12, 1, 2)  # Nov, Dec, Jan, Feb
SOC_THRESHOLDS = (7, 8, 9, 10, 11, 12, 13)


def summarize_days(
    df: pd.DataFrame,
    months: Iterable[int],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Return one row per (year, month) with counts of days hitting SOC thresholds."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month

    day_max = (
        data.groupby("date")
        .agg(max_soc=("battery_soc_kwh", "max"), year=("year", "first"), month=("month", "first"))
        .reset_index(drop=True)
    )
    day_max = day_max[day_max["month"].isin(months)]

    rows: list[dict[str, object]] = []
    for (year, month), group in day_max.groupby(["year", "month"]):
        row = {
            "year": year,
            "month": month,
            "month_name": pd.Timestamp(year=year, month=month, day=1).strftime("%b"),
        }
        for threshold in thresholds:
            row[f"days_gte_{threshold:g}kWh"] = int((group["max_soc"] >= threshold).sum())
        row["sort_key"] = year * 12 + month
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary = summary.sort_values("sort_key").drop(columns="sort_key")
    # Reorder columns for readability.
    ordered_cols = ["year", "month", "month_name"] + [
        c for c in summary.columns if c.startswith("days_gte_")
    ]
    return summary[ordered_cols]


def analyze_variant(name: str, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["timestamp", "battery_soc_kwh"])
    return summarize_days(df, TARGET_MONTHS, SOC_THRESHOLDS)


def parse_thresholds(columns: list[str]) -> list[float]:
    vals: list[float] = []
    for col in columns:
        num = col.replace("days_gte_", "").replace("kWh", "")
        vals.append(float(num))
    return vals


def plot_month_lines(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count_cols = [c for c in summary.columns if c.startswith("days_gte_")]
    thresholds = parse_thresholds(count_cols)

    unique_months = summary[["year", "month", "month_name"]].drop_duplicates()
    month_order = {month: idx for idx, month in enumerate(TARGET_MONTHS)}
    unique_months["order"] = unique_months["month"].map(month_order)
    unique_months = unique_months.sort_values("order")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, (_, row) in zip(axes_flat, unique_months.iterrows()):
        year, month = row["year"], row["month"]
        month_name = row["month_name"]
        month_data = summary[(summary["year"] == year) & (summary["month"] == month)]

        for variant in VARIANT_FILES:
            variant_row = month_data[month_data["variant"] == variant]
            if variant_row.empty:
                continue
            values = [variant_row.iloc[0][col] for col in count_cols]
            ax.plot(thresholds, values, marker="o", label=variant)

        ax.set_title(f"{month_name} {year}")
        ax.set_xlabel("SOC threshold (kWh)")
        ax.set_ylabel("Days with SOC ≥ threshold")
        ax.set_xticks(thresholds, [f"{int(t)}" if t.is_integer() else f"{t:g}" for t in thresholds])
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(unique_months) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Variant", loc="upper center", ncol=len(handles))

    fig.suptitle("Days per Month Reaching Battery SOC Thresholds")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    filename = output_dir / "251114_soc_days.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    combined: list[pd.DataFrame] = []

    for variant, rel_path in VARIANT_FILES.items():
        csv_path = project_root / rel_path
        if not csv_path.exists():
            print(f"[{variant}] Missing file: {csv_path}")
            continue

        summary = analyze_variant(variant, csv_path)
        print(f"\n=== {variant} ({rel_path.name}) ===")
        if summary.empty:
            print("No November–February data found.")
            continue

        summary.insert(0, "variant", variant)
        combined.append(summary)
        print(summary.drop(columns="variant").to_string(index=False))

    if combined:
        all_summary = pd.concat(combined, ignore_index=True)
        figs_dir = project_root / "figs"
        plot_month_lines(all_summary, figs_dir)
        print(f"\nSaved combined monthly plot to {figs_dir}")


if __name__ == "__main__":
    main()
