import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString
import math
import os
import json
import base64
import requests
from io import StringIO
from datetime import datetime, timedelta, timezone

# =========================================================
# 1. CONFIGURATION
# =========================================================
GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO  = os.environ["GH_REPO"]

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=41.3851&longitude=2.1734"
    "&hourly=temperature_2m,cloud_cover,sunshine_duration"
    "&daily=sunrise,sunset"
    "&timezone=Europe%2FMadrid"
    "&forecast_days=7"
)

SUN_ANGLES_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=41.3851&longitude=2.1734"
    "&hourly=is_day"
    "&daily=sunrise,sunset"
    "&timezone=Europe%2FMadrid"
    "&forecast_days=7"
    "&models=best_match"
)

# =========================================================
# 2. LOAD STATIC DATA FROM GITHUB
# =========================================================
print("Loading venues.csv...")
venues_raw = requests.get(
    f"https://raw.githubusercontent.com/{GH_REPO}/main/venues.csv"
).text
venues_df = pd.read_csv(StringIO(venues_raw))

print("Loading buildings.csv from LFS checkout...")
buildings_df = pd.read_csv("buildings.csv")

print(f"Venues: {len(venues_df)} | Buildings: {len(buildings_df)}")

# =========================================================
# 3. CLEAN BUILDINGS
# =========================================================
buildings_df["building_height_m"] = (
    buildings_df["Z_MAX_VOL"] - buildings_df["Z_MIN_VOL"]
)
buildings_df = buildings_df[buildings_df["building_height_m"] > 0].copy()
buildings_df["geometry"] = buildings_df["WKT"].apply(wkt.loads)

buildings_gdf = gpd.GeoDataFrame(
    buildings_df, geometry="geometry", crs="EPSG:4326"
).to_crs(epsg=25831)

print(f"Buildings after cleaning: {len(buildings_gdf)}")

# =========================================================
# 4. CLEAN VENUES
# =========================================================
venue_lat_col = "Latitude"
venue_lon_col = "Longitude"
venue_id_col  = "features.properties.@id"

for col in [venue_lat_col, venue_lon_col]:
    venues_df[col] = pd.to_numeric(
        venues_df[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

venues_df = venues_df.dropna(subset=[venue_lat_col, venue_lon_col])
venues_df = venues_df[
    (venues_df[venue_lon_col] >= 2.05) & (venues_df[venue_lon_col] <= 2.25) &
    (venues_df[venue_lat_col] >= 41.30) & (venues_df[venue_lat_col] <= 41.48)
].copy()

venues_gdf = gpd.GeoDataFrame(
    venues_df,
    geometry=gpd.points_from_xy(venues_df[venue_lon_col], venues_df[venue_lat_col]),
    crs="EPSG:4326"
).to_crs(epsg=25831)

print(f"Venues after Barcelona filter: {len(venues_gdf)}")

# =========================================================
# 5. SPATIAL INDEX + HELPERS
# =========================================================
building_sindex = buildings_gdf.sindex

def azimuth_to_unit_vector(azimuth_deg):
    az_rad = math.radians(azimuth_deg)
    return math.sin(az_rad), math.cos(az_rad)

def make_sun_ray(point, azimuth_deg, max_distance_m=300):
    dx, dy = azimuth_to_unit_vector(azimuth_deg)
    return LineString([
        point,
        Point(point.x + dx * max_distance_m,
              point.y + dy * max_distance_m)
    ])

def is_venue_sunny(venue_point, sun_elevation_deg, sun_azimuth_deg):
    """Returns True if no building blocks the sun for this venue."""
    if sun_elevation_deg <= 0:
        return False  # Sun is below horizon — always dark

    sun_ray = make_sun_ray(venue_point, sun_azimuth_deg)
    possible_idx = list(building_sindex.intersection(sun_ray.bounds))
    candidate_buildings = buildings_gdf.iloc[possible_idx]

    for _, building in candidate_buildings.iterrows():
        if not sun_ray.intersects(building.geometry):
            continue
        distance_m = venue_point.distance(building.geometry)
        if distance_m <= 0:
            continue
        required_height_m = math.tan(math.radians(sun_elevation_deg)) * distance_m
        if float(building["building_height_m"]) >= required_height_m:
            return False  # Blocked

    return True  # No blocker found

# =========================================================
# 6. FETCH 7-DAY HOURLY FORECAST
# =========================================================
print("\nFetching 7-day forecast from Open-Meteo...")
resp = requests.get(FORECAST_URL)
forecast_data = resp.json()

hourly        = forecast_data["hourly"]
times         = hourly["time"]           # e.g. "2026-04-08T00:00"
temperatures  = hourly["temperature_2m"]
cloud_covers  = hourly["cloud_cover"]

# Fetch sun angles using a separate computation approach
# We compute sun elevation/azimuth using astronomical formulas
# since Open-Meteo does not directly provide azimuth in forecast

def compute_sun_angles(dt_utc, lat=41.3851, lon=2.1734):
    """
    Compute sun elevation and azimuth for a given UTC datetime and location.
    Uses standard astronomical formulas.
    Returns (elevation_deg, azimuth_deg)
    """
    # Day of year
    n = dt_utc.timetuple().tm_yday

    # Solar declination
    decl = math.radians(23.45 * math.sin(math.radians(360/365 * (n - 81))))

    # Equation of time (minutes)
    B    = math.radians(360/365 * (n - 81))
    eot  = 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    # Solar noon correction
    lon_correction = lon / 15.0  # hours
    solar_time     = dt_utc.hour + dt_utc.minute / 60.0 + lon_correction + eot / 60.0
    hour_angle_deg = 15.0 * (solar_time - 12.0)
    hour_angle     = math.radians(hour_angle_deg)

    lat_rad = math.radians(lat)

    # Elevation
    sin_elev = (math.sin(lat_rad) * math.sin(decl) +
                math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    elevation = math.degrees(math.asin(max(-1, min(1, sin_elev))))

    # Azimuth
    cos_az = ((math.sin(decl) - math.sin(lat_rad) * sin_elev) /
              (math.cos(lat_rad) * math.cos(math.radians(elevation))))
    cos_az  = max(-1, min(1, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if hour_angle_deg > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

print(f"Got {len(times)} hourly slots over 7 days")

# =========================================================
# 7. GROUP BY DAY AND PROCESS
# =========================================================
# Group hours by date
from collections import defaultdict
days = defaultdict(list)
for i, t in enumerate(times):
    date_str = t[:10]  # "2026-04-08"
    days[date_str].append({
        "datetime_local": t,
        "temperature_c":  temperatures[i],
        "cloud_cover":    cloud_covers[i],
        "slot_index":     i
    })

print(f"\nProcessing {len(days)} days × {len(venues_gdf)} venues...")

# =========================================================
# 8. FOR EACH DAY: CALCULATE SHADOW FOR ALL VENUES ALL HOURS
# =========================================================
for day_str, hours in sorted(days.items()):
    print(f"\n--- {day_str} ({len(hours)} hours) ---")
    day_results = []

    for hour_data in hours:
        dt_local_str = hour_data["datetime_local"]  # "2026-04-08T14:00"
        temp         = hour_data["temperature_c"]
        cloud        = hour_data["cloud_cover"]

        # Parse as local Madrid time, convert to UTC for sun angle calc
        dt_local = datetime.strptime(dt_local_str, "%Y-%m-%dT%H:%M")
        # Madrid is UTC+1 in winter, UTC+2 in summer (approx)
        # Simple approach: subtract 2 hours for summer, 1 for winter
        month = dt_local.month
        utc_offset = 2 if 3 <= month <= 10 else 1
        dt_utc = dt_local - timedelta(hours=utc_offset)

        # Compute sun angles
        sun_elev, sun_az = compute_sun_angles(dt_utc)

        # Skip nighttime hours (sun below horizon)
