import nfl_data_py as nfl
import pandas as pd
import numpy as np

from position_group import pos_group

# Snap counts: aggregate to team-week-position group
snaps = nfl.import_snap_counts([2021, 2022, 2023, 2024])
# Map granular positions to broader position groups
snaps["pos_group"] = snaps["position"].apply(pos_group)
# Total snaps = offense + defense 
snaps["total_snaps"] = snaps.get("offense_snaps", 0) + snaps.get("defense_snaps", 0)

snaps_g = (
    snaps.groupby(["season","week","team","pos_group"], as_index=False)["total_snaps"]
         .sum()
         .rename(columns={"total_snaps":"snaps"})
)
snaps_g = snaps_g[snaps_g["pos_group"].isin(["QB","OL","DL","LB","DB","SKILL"])]

# Injuries: aggregate to team-week-position group
# Injury event = "Out" or "Doubtful" designation in the weekly report
inj = nfl.import_injuries([2021, 2022, 2023, 2024])
inj["pos_group"] = inj["position"].apply(pos_group)

status_col = "report_status"
inj[status_col] = inj[status_col].astype(str)

inj["inj_event"] = inj[status_col].isin(["Out","Doubtful"]).astype(int)

inj_g = (
    inj.groupby(["season","week","team","pos_group"], as_index=False)["inj_event"]
       .sum()
       .rename(columns={"inj_event":"injuries"})
)
inj_g = inj_g[inj_g["pos_group"].isin(["QB","OL","DL","LB","DB","SKILL"])]



# Save outputs so other files can load quickly
print(snaps_g.head())
print(inj_g.head())

snaps_g.to_parquet("snaps_g_2021_2024.parquet", index=False)
inj_g.to_parquet("inj_g_2021_2024.parquet", index=False)
