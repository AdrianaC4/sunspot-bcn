import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString
import math
import os
import base64
import requests
from io import StringIO
import json

# =========================================================
# 1. CONFIGURATION
# =========================================================
SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO  = os.environ["GH_REPO"]

SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    f"/export?format=csv&gid=881521698"
)

# =========================================================
# 2. LOAD DATA
# =========================================================
print("Loading venues.csv from local checkout...")
venues_df = pd.read_csv("venues.csv")

print("Loading buildings.csv from local checkout (LFS)...")
buildings_df = pd.read_csv("buildings.csv")

print("Loading sun context from Google Sheets...")
response = requests.get(SHEET_URL, allow_redirects=True)
sun_df = pd.read_csv(StringIO(response.text))

print("Buildings rows:", len(buildings_df))
print("Venues rows:",    len(venues_df))
print("Sun rows:",       len(sun_df))

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

print("Buildings after cleaning:", len(buildings_gdf))

# =========================================================
# 4. CLEAN VENUES
# =========================================================
venue_lat_col = "Latitude"
venue_lon_col = "Longitude"
venue_id_col  = "features.properties.@id"

for col in [venue_lat_col, venue_lon_col]:
    venues_df[col] = (
        pd.to_numeric(
            venues_df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
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

print("Venues after Barcelona filter:", len(venues_gdf))

# =========================================================
# 5. CLEAN SUN CONTEXT
# =========================================================
for col in ["sun_elevation_deg", "sun_azimuth_deg", "cloud_cover"]:
    sun_df[col] = pd.to_numeric(
        sun_df[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

sun_df["timestamp_utc"] = pd.to_datetime(sun_df["timestamp_utc"])
latest_sun        = sun_df.sort_values("timestamp_utc").iloc[-1]
sun_elevation_deg = float(latest_sun["sun_elevation_deg"])
sun_azimuth_deg   = float(latest_sun["sun_azimuth_deg"])
cloud_cover       = float(latest_sun["cloud_cover"])

print("Sun elevation:", sun_elevation_deg)
print("Sun azimuth:",   sun_azimuth_deg)
print("Cloud cover:",   cloud_cover)
print("Timestamp UTC:", latest_sun["timestamp_utc"])

# =========================================================
# 6. SPATIAL INDEX + HELPERS
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

# =========================================================
# 7. SHADOW CALCULATION
# =========================================================
results      = []
total_venues = len(venues_gdf)
print(f"\nProcessing {total_venues} venues...")

for i, (_, venue) in enumerate(venues_gdf.iterrows(), start=1):
    if i % 100 == 0 or i == 1 or i == total_venues:
        print(f"  Venue {i}/{total_venues}")

    venue_id       = venue[venue_id_col]
    venue_name     = venue.get("Amenity_name", None)
    amenity_type   = venue.get("Amenity_type", None)
    terrace_status = venue.get("Terrace_Status", None)
    address        = venue.get("Address", None)
    latitude       = venue[venue_lat_col] if venue_lat_col in venue.index else None
    longitude      = venue[venue_lon_col] if venue_lon_col in venue.index else None
    venue_point    = venue.geometry
    sun_ray        = make_sun_ray(venue_point, sun_azimuth_deg)

    possible_idx        = list(building_sindex.intersection(sun_ray.bounds))
    candidate_buildings = buildings_gdf.iloc[possible_idx]
    blocking_buildings  = []

    for _, building in candidate_buildings.iterrows():
        if not sun_ray.intersects(building.geometry):
            continue
        distance_m = venue_point.distance(building.geometry)
        if distance_m <= 0:
            continue
        required_height_m = math.tan(math.radians(sun_elevation_deg)) * distance_m
        if float(building["building_height_m"]) >= required_height_m:
            blocking_buildings.append({
                "building_id":       building["FID"],
                "distance_m":        distance_m,
                "required_height_m": required_height_m,
                "building_height_m": float(building["building_height_m"])
            })

    if blocking_buildings:
        nearest = sorted(blocking_buildings, key=lambda x: x["distance_m"])[0]
        results.append({
            "venue_id":            venue_id,
            "venue_name":          venue_name,
            "amenity_type":        amenity_type,
            "terrace_status":      terrace_status,
            "address":             address,
            "latitude":            latitude,
            "longitude":           longitude,
            "is_sunny_now":        0,
            "blocker_building_id": nearest["building_id"],
            "blocker_distance_m":  nearest["distance_m"],
            "required_height_m":   nearest["required_height_m"],
            "blocker_height_m":    nearest["building_height_m"],
            "sun_elevation_deg":   sun_elevation_deg,
            "sun_azimuth_deg":     sun_azimuth_deg,
            "cloud_cover":         cloud_cover,
            "last_updated":        str(latest_sun["timestamp_utc"])
        })
    else:
        results.append({
            "venue_id":            venue_id,
            "venue_name":          venue_name,
            "amenity_type":        amenity_type,
            "terrace_status":      terrace_status,
            "address":             address,
            "latitude":            latitude,
            "longitude":           longitude,
            "is_sunny_now":        1,
            "blocker_building_id": None,
            "blocker_distance_m":  None,
            "required_height_m":   None,
            "blocker_height_m":    None,
            "sun_elevation_deg":   sun_elevation_deg,
            "sun_azimuth_deg":     sun_azimuth_deg,
            "cloud_cover":         cloud_cover,
            "last_updated":        str(latest_sun["timestamp_utc"])
        })

# =========================================================
# 8. PUSH RESULTS TO GITHUB
# =========================================================
print("\nPushing venue_sun_results.csv to GitHub...")

results_df  = pd.DataFrame(results)
csv_content = results_df.to_csv(index=False)
encoded     = base64.b64encode(csv_content.encode()).decode()

api_url = f"https://api.github.com/repos/{GH_REPO}/contents/venue_sun_results.csv"
headers = {
    "Authorization": f"token {GH_TOKEN}",
    "Accept":        "application/vnd.github.v3+json"
}

get_resp = requests.get(api_url, headers=headers)
sha      = get_resp.json().get("sha") if get_resp.status_code == 200 else None

payload = {
    "message": f"Auto-update venue_sun_results [{latest_sun['timestamp_utc']}]",
    "content": encoded
}
if sha:
    payload["sha"] = sha

put_resp = requests.put(api_url, headers=headers, data=json.dumps(payload))

if put_resp.status_code in [200, 201]:
    print("✅ venue_sun_results.csv pushed to GitHub successfully!")
else:
    print("❌ GitHub push failed:", put_resp.status_code, put_resp.text)

print("\n=== DONE ===")
print(results_df["is_sunny_now"].value_counts(dropna=False))
