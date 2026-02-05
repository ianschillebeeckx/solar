#!/usr/bin/env python3
"""
Master driver for the 251107 analysis pipeline.

When you obtain additional historical utility data (e.g., a new PG&E export
covering later months), update the `USAGE_2025_FP` path and any related inputs
inside `251107_interpolate_2025_usage.py`, then rerun this script to regenerate
all downstream artifacts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Scripts must remain in the order they were copied/derived so that each stage
# feeds its outputs into the next.
SCRIPTS = [
    ("Interpolate 2025 usage", "251107_interpolate_2025_usage.py", {"MPLBACKEND": "Agg"}),
    ("Therm → kWh adjustment", "251107_therm2kWh.py", None),
    ("Battery simulations", "251107_sim_batteries.py", None),
    ("Cost estimation", "251107_est_cost.py", None),
    ("Payback visualization", "251107_payback.py", {"MPLBACKEND": "Agg"}),
]


def run_script(description: str, script_name: str, extra_env: dict[str, str] | None) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Expected script missing: {script_path}")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"\n=== Running: {description} ({script_name}) ===")
    subprocess.run([sys.executable, str(script_path)], check=True, env=env, cwd=str(BASE_DIR))


def main() -> None:
    for description, script_name, extra_env in SCRIPTS:
        run_script(description, script_name, extra_env)
    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
