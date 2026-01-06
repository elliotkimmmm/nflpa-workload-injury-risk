import nfl_data_py as nfl
import pandas as pd
import numpy as np
import pytz
from math import radians, sin, cos, sqrt, atan2

# Team home stadium coordinates + home time zone (used to approximate travel + tz shifts)
team_home = {
    # AFC East
    "BUF": {"lat": 42.7738, "lon": -78.7868, "tz": "US/Eastern"},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "tz": "US/Eastern"},
    "NE":  {"lat": 42.0909, "lon": -71.2643, "tz": "US/Eastern"},
    "NYJ": {"lat": 40.8135, "lon": -74.0745, "tz": "US/Eastern"},
    # AFC North
    "BAL": {"lat": 39.2780, "lon": -76.6227, "tz": "US/Eastern"},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "tz": "US/Eastern"},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "tz": "US/Eastern"},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "tz": "US/Eastern"},
    # AFC South
    "HOU": {"lat": 29.6847, "lon": -95.4107, "tz": "US/Central"},
    "IND": {"lat": 39.7601, "lon": -86.1639, "tz": "US/Eastern"},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "tz": "US/Eastern"},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "tz": "US/Central"},
    # AFC West
    "DEN": {"lat": 39.7439, "lon": -105.0201, "tz": "US/Mountain"},
    "KC":  {"lat": 39.0489, "lon": -94.4839, "tz": "US/Central"},
    "LV":  {"lat": 36.0908, "lon": -115.1830, "tz": "US/Pacific"},
    "LAC": {"lat": 33.9535, "lon": -118.3392, "tz": "US/Pacific"},
    # NFC East
    "DAL": {"lat": 32.7473, "lon": -97.0945, "tz": "US/Central"},
    "NYG": {"lat": 40.8135, "lon": -74.0745, "tz": "US/Eastern"},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "tz": "US/Eastern"},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "tz": "US/Eastern"},
    # NFC North
    "CHI": {"lat": 41.8623, "lon": -87.6167, "tz": "US/Central"},
    "DET": {"lat": 42.3400, "lon": -83.0456, "tz": "US/Eastern"},
    "GB":  {"lat": 44.5013, "lon": -88.0622, "tz": "US/Central"},
    "MIN": {"lat": 44.9738, "lon": -93.2581, "tz": "US/Central"},
    # NFC South
    "ATL": {"lat": 33.7554, "lon": -84.4008, "tz": "US/Eastern"},
    "CAR": {"lat": 35.2258, "lon": -80.8528, "tz": "US/Eastern"},
    "NO":  {"lat": 29.9511, "lon": -90.0812, "tz": "US/Central"},
    "TB":  {"lat": 27.9759, "lon": -82.5033, "tz": "US/Eastern"},
    # NFC West
    "ARI": {"lat": 33.5277, "lon": -112.2626, "tz": "US/Arizona"},
    "LAR": {"lat": 33.9535, "lon": -118.3392, "tz": "US/Pacific"},
    "SF":  {"lat": 37.4030, "lon": -121.9700, "tz": "US/Pacific"},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "tz": "US/Pacific"},
}

sched = nfl.import_schedules([2021, 2022, 2023, 2024])
#print(sched.shape)

core = [
    "game_id","season","week","gameday","gametime","weekday",
    "home_team","away_team",
    "home_rest","away_rest",
    "stadium","location","roof","surface","temp","wind","div_game"
]
core = [c for c in core if c in sched.columns]
games = sched[core].drop_duplicates("game_id").copy()

# Parse date/time fields
games["gameday"] = pd.to_datetime(games["gameday"])
games["kickoff_dt"] = pd.to_datetime(
    games["gameday"].dt.strftime("%Y-%m-%d") + " " + games["gametime"].astype(str),
    errors="coerce"
)

# Expand to team-week - two rows per game: one for each team
home_rows = games.assign(
    team=games["home_team"],
    opponent=games["away_team"],
    is_home=1,
    rest_days=games["home_rest"]
)
away_rows = games.assign(
    team=games["away_team"],
    opponent=games["home_team"],
    is_home=0,
    rest_days=games["away_rest"]
)

team_week = (
    pd.concat([home_rows, away_rows], ignore_index=True)
    .sort_values(["team","week"])
    .reset_index(drop=True)
)
team_week["is_away"] = 1 - team_week["is_home"]

# Construct rest-based stressors - bye proxy, opponent bye proxy, rest differential
# Coming off bye: long rest week (commonly >= 13 days)
team_week["coming_off_bye"] = (team_week["rest_days"] >= 13).astype(int)

opp_rest = team_week[["season","week","team","rest_days","coming_off_bye"]].rename(
    columns={"team":"opponent", "rest_days":"opp_rest_days", "coming_off_bye":"opp_coming_off_bye"}
)
team_week = team_week.merge(opp_rest, on=["season","week","opponent"], how="left")
team_week["opp_rest_days"] = team_week["opp_rest_days"].fillna(0)
team_week["opp_coming_off_bye"] = team_week["opp_coming_off_bye"].fillna(0).astype(int)

# Positive rest_diff means the opponent had more rest (rest disadvantage for the team)
team_week["rest_diff"] = team_week["opp_rest_days"] - team_week["rest_days"]

# Assign game location coordinates + time zone
# Default: use the home team's stadium. Override for international/neutral sites
team_week["game_lat"] = team_week["home_team"].map(lambda t: team_home.get(t, {}).get("lat"))
team_week["game_lon"] = team_week["home_team"].map(lambda t: team_home.get(t, {}).get("lon"))
team_week["game_tz"]  = team_week["home_team"].map(lambda t: team_home.get(t, {}).get("tz"))

intl_sites = [
    ("London",  51.5074,  -0.1278,  "Europe/London"),
    ("Munich",  48.1351,  11.5820,  "Europe/Berlin"),
    ("Frankfurt", 50.1109, 8.6821,  "Europe/Berlin"),
    ("Berlin",  52.5200,  13.4050,  "Europe/Berlin"),
    ("Mexico",  19.4326, -99.1332,  "America/Mexico_City"),
    ("Sao",    -23.5505, -46.6333,  "America/Sao_Paulo"),   # São Paulo
    ("Brazil", -23.5505, -46.6333,  "America/Sao_Paulo"),
    ("Germany", 50.1109, 8.6821,    "Europe/Berlin"),
]

loc_str = team_week["location"].astype(str)

for key, lat, lon, tz in intl_sites:
    mask = loc_str.str.contains(key, case=False, na=False)
    team_week.loc[mask, ["game_lat","game_lon","game_tz"]] = [lat, lon, tz]

# Compute travel miles and time zone shifts relative to the team's previous game
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points (Haversine formula)."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

team_week = team_week.sort_values(["team","season","week"]).reset_index(drop=True)

team_week["prev_game_lat"] = team_week.groupby("team")["game_lat"].shift(1)
team_week["prev_game_lon"] = team_week.groupby("team")["game_lon"].shift(1)

team_week["travel_miles"] = team_week.apply(
    lambda r: haversine(r["prev_game_lat"], r["prev_game_lon"], r["game_lat"], r["game_lon"])
    if pd.notna(r["prev_game_lat"]) and pd.notna(r["game_lat"]) else 0.0,
    axis=1
)

def tz_offset_hours(tzname: str) -> float:
    """Return UTC offset (hours) for a time zone at a reference datetime (handles DST)."""
    try:
        return pytz.timezone(tzname).utcoffset(pd.Timestamp("2023-10-01")).total_seconds() / 3600
    except Exception:
        return np.nan

team_week["tz_offset"] = team_week["game_tz"].apply(tz_offset_hours)
team_week["prev_tz_offset"] = team_week.groupby("team")["tz_offset"].shift(1)

# Absolute change in time zone (hours) from prior game; week 1 has shift = 0
team_week["timezone_shift"] = (team_week["tz_offset"] - team_week["prev_tz_offset"]).abs().fillna(0.0)

# Quick QA checks + save for downstream analysis
print("Missing game coords:", team_week["game_lat"].isna().mean())
print("Missing time zones:", team_week["game_tz"].isna().mean())
print(team_week[["team","season","week","home_team","away_team","location","travel_miles","timezone_shift"]].head())

team_week.to_parquet("team_week_2021_2024.parquet", index=False)