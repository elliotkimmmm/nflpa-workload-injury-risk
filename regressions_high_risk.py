import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# Goal: Estimate position-specific regression models linking schedule
# stressors to injury risk (injuries per 1,000 snaps), with
# team + week fixed effects and team-clustered standard errors.

team_week = pd.read_parquet("team_week_2021_2024.parquet")
snaps_g   = pd.read_parquet("snaps_g_2021_2024.parquet")
inj_g     = pd.read_parquet("inj_g_2021_2024.parquet")

# Rebuild team-week-position panel
pos_groups = ["SKILL", "LB", "DB"]

# Build a team-week-position panel via a cross join (one row per team-week-pos_group)
team_week_pos = (
    team_week
    .assign(key=1)
    .merge(pd.DataFrame({"pos_group": pos_groups, "key": 1}), on="key")
    .drop("key", axis=1)
)

# Merge in snaps and injury counts for each team-week-position group
team_week_pos = team_week_pos.merge(
    snaps_g, on=["season","week","team","pos_group"], how="left"
)

team_week_pos = team_week_pos.merge(
    inj_g, on=["season","week","team","pos_group"], how="left"
)

# Replace missing counts with zeros (some team-week-pos combinations have no snaps/injuries)
team_week_pos["snaps"] = team_week_pos["snaps"].fillna(0)
team_week_pos["injuries"] = team_week_pos["injuries"].fillna(0)

# Injury rate per 1,000 snaps (undefined when snaps == 0)
team_week_pos["injury_rate_per_1000"] = np.where(
    team_week_pos["snaps"] > 0,
    1000 * team_week_pos["injuries"] / team_week_pos["snaps"],
    np.nan
)

# Drop Week 1
team_week_pos = team_week_pos[team_week_pos["week"] > 1]
# Scaled version of travel miles
team_week_pos["travel_1000"] = team_week_pos["travel_miles"] / 1000


# Run regressions by position
results = {}

for pos in pos_groups:
    # Restrict to weeks where the position group has positive snap exposure
    df = team_week_pos[
        (team_week_pos["pos_group"] == pos) &
        (team_week_pos["snaps"] > 0)
    ].copy()

    # Linear model: injury rate on schedule stressors with fixed effects
    model = smf.ols(
        formula="""
        injury_rate_per_1000 ~
        travel_1000 +
        timezone_shift +
        rest_diff +
        is_away +
        C(team) +
        C(week)
        """,
        data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["team"]})

    results[pos] = model
    print("\n" + "="*60)
    print(f"POSITION GROUP: {pos}")
    print("="*60)
    print(model.summary().as_text(), flush = True)

# Extract coefficients + 95% confidence intervals for key stressors
def extract_coefs(model, position):
    rows = []
    for var in ["travel_1000", "timezone_shift", "rest_diff", "is_away"]:
        coef = model.params[var]
        se   = model.bse[var]
        rows.append({
            "position": position,
            "variable": var,
            "coef": coef,
            "lower": coef - 1.96 * se,
            "upper": coef + 1.96 * se
        })
    return pd.DataFrame(rows)

coef_dfs = []

for pos in ["SKILL", "LB", "DB"]:
    model = results[pos]
    coef_dfs.append(extract_coefs(model, pos))

coef_df = pd.concat(coef_dfs, ignore_index=True)

# Plot coefficient estimates with 95% confidence intervals by position group
def plot_position(coef_df, position, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    df = coef_df[coef_df["position"] == position].copy()

    plt.figure(figsize=(7,4))
    plt.errorbar(
        df["coef"],
        df["variable"],
        xerr=[df["coef"] - df["lower"], df["upper"] - df["coef"]],
        fmt="o",
        capsize=4
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(f"{position}: Effect of Schedule Stressors on Injury Risk")
    plt.xlabel("Change in injuries per 1,000 snaps")
    plt.ylabel("")
    plt.tight_layout()

    outpath = os.path.join(outdir, f"coef_{position}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")

for pos in ["SKILL", "LB", "DB"]:
    plot_position(coef_df, pos)
