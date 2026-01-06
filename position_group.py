import pandas as pd

def pos_group(pos: str) -> str:
    """
    Map granular NFL position labels to broad position groups used in analysis.

    Groups:
        QB     -> Quarterback
        OL     -> Offensive Line
        DL     -> Defensive Line
        LB     -> Linebacker
        DB     -> Defensive Back
        SKILL  -> RB / WR / TE / FB
        OTHER  -> All remaining or missing positions
    """
    if pd.isna(pos):
        return "OTHER"
    pos = str(pos).upper()
    if pos == "QB":
        return "QB"
    if pos in {"C","G","T","OL"}:
        return "OL"
    if pos in {"DT","NT","DE","DL"}:
        return "DL"
    if pos in {"LB","ILB","OLB"}:
        return "LB"
    if pos in {"CB","DB","S","FS","SS"}:
        return "DB"
    if pos in {"RB","WR","TE","FB"}:
        return "SKILL"
    return "OTHER"