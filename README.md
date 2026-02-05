# solar

Solar payback analysis for residential PV + battery installation.

## Current Analysis

Run the 260204 pipeline for the actual 6.6kW install:
```bash
cd src && python 260204_run_pipeline.py
```

## Notes

**Historical scripts (251107_*, 250xxx_*) may be broken.** PII (address, account numbers, coordinates) was removed from input files, which changed the row offsets. The `skiprows` parameter in older scripts still references the original file structure.

Only the 260204_* scripts have been updated to work with the cleaned input files.