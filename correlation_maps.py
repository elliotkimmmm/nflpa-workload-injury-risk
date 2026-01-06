import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a position-group heat map of Pearson correlations between
# schedule stressors and injury rate (injuries per 1,000 snaps).
# Inputs are preprocessed parquet files generated earlier.

# Expand schedule to team-week-position group (cross join)
pos_groups = pd.DataFrame({
    "pos_group": ["QB","OL","DL","LB","DB","SKILL"]
})

# Load preprocessed datasets
team_week = pd.read_parquet("team_week_2021_2024.parquet")
snaps_g   = pd.read_parquet("snaps_g_2021_2024.parquet")
inj_g     = pd.read_parquet("inj_g_2021_2024.parquet")


team_week_pos = team_week.merge(pos_groups, how="cross")

# Merge in snaps and injuries by team-week-position group
team_week_pos = team_week_pos.merge(
    snaps_g,
    on=["season","week","team","pos_group"],
    how="left"
)

team_week_pos = team_week_pos.merge(
    inj_g,
    on=["season","week","team","pos_group"],
    how="left"
)

# Drop week 1 since prior-game travel/timezone shift are undefined by construction
team_week_pos = team_week_pos[team_week_pos["week"] > 1]

# Fill missing counts (teams can have zero snaps or zero injuries for a pos group in a given week)
team_week_pos["snaps"] = team_week_pos["snaps"].fillna(0)
team_week_pos["injuries"] = team_week_pos["injuries"].fillna(0)

# Injury rate per 1,000 snaps (undefined when snaps == 0)
team_week_pos["injury_rate_per_1000"] = np.where(
    team_week_pos["snaps"] > 0,
    1000 * team_week_pos["injuries"] / team_week_pos["snaps"],
    np.nan
)

stressors = [
    "rest_diff",
    "opp_coming_off_bye",
    "is_away",
    "travel_miles",
    "timezone_shift"
]

corr_rows = []

for pos in team_week_pos["pos_group"].unique():
    df = team_week_pos[
        (team_week_pos["pos_group"] == pos) &
        (team_week_pos["snaps"] > 0)
    ]
    
    for var in stressors:
        corr = df[[var, "injury_rate_per_1000"]].corr().iloc[0,1]
        corr_rows.append({
            "pos_group": pos,
            "variable": var,
            "correlation": corr
        })

# Keep only observations with defined injury rates for correlation computation
corr_df = pd.DataFrame(corr_rows)

corr_map = corr_df.pivot(
    index="pos_group",
    columns="variable",
    values="correlation"
)

print(corr_map)

plt.figure(figsize=(8, 4))
sns.heatmap(
    corr_map,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Injury Rate Correlations by Position Group (2021-2024)")
plt.ylabel("Position Group")
plt.xlabel("")
plt.tight_layout()
plt.show()