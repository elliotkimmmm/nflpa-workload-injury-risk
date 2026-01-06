# NFLPA Analytics Case Competition – Cumulative Workload & Injury Risk

This repository contains all code and intermediate datasets used in the analysis of
schedule-related workload stressors and NFL player injury risk (2021–2024).

## Structure
- `schedule.py`: constructs team-week schedule stressors (travel, rest, time zones)
- `snaps.py`: aggregates snap counts by position group
- `injuries.py`: processes injury reports and classifies injury types
- `correlation_maps.py`: generates position-specific correlation heat maps
- `regressions_high_risk.py`: estimates fixed-effects regression models and plots results
- `plots/`: exported figures used in the paper

## How to run
Install required packages:
pip install nfl_data_py pandas numpy matplotlib seaborn statsmodels pytz

Run scripts in order:
1. schedule.py
2. snaps.py
3. injuries.py
4. correlation_maps.py
5. regressions_high_risk.py

All data sources are publicly available via the `nfl_data_py` package.
