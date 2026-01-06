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

All data sources are publicly available via the `nfl_data_py` package.