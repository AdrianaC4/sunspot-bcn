# Copyright (c) 2026 Adriana Cavallaro. All rights reserved.
# SunSpot BCN — https://adrianac4.github.io/sunspot-bcn/

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
from collections import defaultdict

# =========================================================
# 1. CONFIGURATION
# =========================================================
GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO  = os.environ["GH_REPO"]

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=41.3851&longitude=2.1734"
    "&hourly=temperature_2m,cloud_cover"
    "&timezone=Europe%2FMadrid"
    "&forecast_days=7"
)

# =========================================================
# 2. LOAD STATIC DATA FROM LOCAL CHECKOUT
# =========================================================
print("Loading venues.csv from local checkout...")
venues_df = pd.read_csv("venues.csv")

print("Loading buildings.csv from local checkout (LFS)...")
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

def compute_sun_angles(dt_utc, lat=41.3851, lon=2.1734):
    """
    Compute sun elevation and azimuth for a given UTC datetime.
    Uses standard astronomical formulas.
    Returns (elevation_deg, azimuth_deg)
    """
    n    = dt_utc.timetuple().tm_yday
    decl = math.radians(23.45 * math.sin(math.radians(360/365 * (n - 81))))
    B    = math.radians(360/365 * (n - 81))
    eot  = 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    lon_correction = lon / 15.0
    solar_time     = dt_utc.hour + dt_utc.minute / 60.0 + lon_correction + eot / 60.0
    hour_angle_deg = 15.0 * (solar_time - 12.0)
    hour_angle     = math.radians(hour_angle_deg)
    lat_rad        = math.radians(lat)

    sin_elev  = (math.sin(lat_rad) * math.sin(decl) +
                 math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    elevation = math.degrees(math.asin(max(-1, min(1, sin_elev))))

    if elevation <= 0:
        return elevation, 0.0

    cos_az  = ((math.sin(decl) - math.sin(lat_rad) * sin_elev) /
               (math.cos(lat_rad) * math.cos(math.radians(elevation))))
    cos_az  = max(-1, min(1, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if hour_angle_deg > 0:
        azimuth = 360 - azimuth

    return elevation, azimuth

def is_venue_sunny(venue_point, sun_elevation_deg, sun_azimuth_deg):
    """Returns True if no building blocks the sun for this venue."""
    if sun_elevation_deg <= 0:
        return False

    sun_ray      = make_sun_ray(venue_point, sun_azimuth_deg)
    possible_idx = list(building_sindex.intersection(sun_ray.bounds))
    candidates   = buildings_gdf.iloc[possible_idx]

    for _, building in candidates.iterrows():
        if not sun_ray.intersects(building.geometry):
            continue
        distance_m = venue_point.distance(building.geometry)
        if distance_m <= 0:
            continue
        required_h = math.tan(math.radians(sun_elevation_deg)) * distance_m
        if float(building["building_height_m"]) >= required_h:
            return False

    return True

# =========================================================
# 6. FETCH 7-DAY HOURLY FORECAST FROM OPEN-METEO
# =========================================================
print("\nFetching 7-day forecast from Open-Meteo...")
resp          = requests.get(FORECAST_URL)
forecast_data = resp.json()
hourly        = forecast_data["hourly"]
times         = hourly["time"]
temperatures  = hourly["temperature_2m"]
cloud_covers  = hourly["cloud_cover"]
print(f"Got {len(times)} hourly slots")

# =========================================================
# 7. GROUP HOURLY SLOTS BY DAY
# =========================================================
days = defaultdict(list)
for i, t in enumerate(times):
    date_str = t[:10]
    days[date_str].append({
        "datetime_local": t,
        "temperature_c":  temperatures[i],
        "cloud_cover":    cloud_covers[i],
    })

# =========================================================
# 8. GITHUB PUSH HELPER
# =========================================================
gh_headers = {
    "Authorization": f"Bearer {GH_TOKEN}",
    "Accept":        "application/vnd.github.v3+json"
}

def push_file_to_github(filename, csv_content, message):
    encoded  = base64.b64encode(csv_content.encode()).decode()
    api_url  = f"https://api.github.com/repos/{GH_REPO}/contents/{filename}"
    get_resp = requests.get(api_url, headers=gh_headers)
    sha      = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    payload  = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    put_resp = requests.put(api_url, headers=gh_headers, data=json.dumps(payload))
    if put_resp.status_code in [200, 201]:
        print(f"  ✅ {filename} pushed")
    else:
        print(f"  ❌ {filename} failed: {put_resp.status_code} {put_resp.text[:200]}")

# =========================================================
# 9. PROCESS EACH DAY
# =========================================================
total_venues = len(venues_gdf)
print(f"\nProcessing {len(days)} days × {total_venues} venues...")

for day_str, hours in sorted(days.items()):
    print(f"\n--- {day_str} ---")
    day_results = []

    for hour_data in hours:
        dt_local_str = hour_data["datetime_local"]
        temp         = hour_data["temperature_c"]
        cloud        = hour_data["cloud_cover"]

        # Convert local Madrid time to UTC
        dt_local   = datetime.strptime(dt_local_str, "%Y-%m-%dT%H:%M")
        month      = dt_local.month
        utc_offset = 2 if 3 <= month <= 10 else 1
        dt_utc     = dt_local - timedelta(hours=utc_offset)

        # Compute sun angles
        sun_elev, sun_az = compute_sun_angles(dt_utc)

        # Skip nighttime
        if sun_elev <= 0:
            continue

        # Process all venues for this hour
        for _, venue in venues_gdf.iterrows():
            venue_point    = venue.geometry
            sunny          = is_venue_sunny(venue_point, sun_elev, sun_az)

            day_results.append({
                "hour":           dt_local_str,
                "venue_id":       venue[venue_id_col],
                "venue_name":     venue.get("Amenity_name", ""),
                "amenity_type":   venue.get("Amenity_type", ""),
                "terrace_status": venue.get("Terrace_Status", ""),
                "address":        venue.get("Address", ""),
                "latitude":       venue[venue_lat_col],
                "longitude":      venue[venue_lon_col],
                "is_sunny":       1 if sunny else 0,
                "sun_elevation":  round(sun_elev, 2),
                "sun_azimuth":    round(sun_az, 2),
                "cloud_cover":    cloud,
                "temperature_c":  temp,
            })

    if not day_results:
        print(f"  No daytime hours, skipping.")
        continue

    df          = pd.DataFrame(day_results)
    csv_content = df.to_csv(index=False)
    sunny_count = df["is_sunny"].sum()
    print(f"  {len(df)} rows | {sunny_count} sunny predictions")
    push_file_to_github(
        f"forecast_{day_str}.csv",
        csv_content,
        f"Forecast update: {day_str}"
    )

print("\n=== FORECAST RUNNER COMPLETE ===")
