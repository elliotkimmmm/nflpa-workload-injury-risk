import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from position_group import pos_group

inj = nfl.import_injuries([2021, 2022, 2023, 2024])
inj["pos_group"] = inj["position"].apply(pos_group)

status_col = "report_status"
inj[status_col] = inj[status_col].astype(str)

# Injury designations that do not represent actual injuries
NON_INJURY = [
    "None",
    "Reserve/COVID activation",
    "COVID Protocols"
]

# Clean NFL injury report data
inj_clean = inj[~inj["report_primary_injury"].isin(NON_INJURY)].copy()
inj_clean = inj_clean.dropna(subset=["report_primary_injury"])

inj_clean["inj_norm"] = (
    inj_clean["report_primary_injury"]
    .str.lower()
    .str.strip()
)

# Bucket injuries into broad anatomical groups
def injury_bucket(x):
    if "knee" in x:
        return "Knee"
    if "ankle" in x or "foot" in x or "toe" in x or "heel" in x:
        return "Foot/Ankle"
    if "hamstring" in x or "groin" in x or "quad" in x or "calf" in x:
        return "Soft Tissue (Lower)"
    if "shoulder" in x or "clavicle" in x or "arm" in x:
        return "Shoulder/Arm"
    if "elbow" in x or "wrist" in x or "hand" in x or "finger" in x:
        return "Upper Extremity (Distal)"
    if "concussion" in x or "head" in x:
        return "Head"
    if "back" in x or "spine" in x or "neck" in x:
        return "Back/Neck"
    return "Other"

inj_clean["injury_group"] = inj_clean["inj_norm"].apply(injury_bucket)

# Aggregate injury counts and visualize top categories
body_part_counts = (
    inj_clean["injury_group"]
    .value_counts()
)
top5 = body_part_counts.head(5)


plt.figure(figsize=(4,4))
plt.bar(top5.index.astype(str), top5.values)
plt.title("NFL Injuries 2021-24")
plt.ylabel("Count")
plt.xlabel("Body Part")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
