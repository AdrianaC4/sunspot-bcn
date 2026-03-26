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
    f"/gviz/tq?tqx=out:csv&sheet=sun_context_barcelona"
)

# =========================================================
# 2. LOAD DATA
# =========================================================
print("Loading venues.csv from local checkout...")
venues_df = pd.read_csv("venues.csv")

print("Loading buildings.csv from local checkout (LFS)...")
buildings_df = pd.read_csv("buildings.csv")

print("Loading sun context from Google Sheets...")
sun_raw = requests.get(SHEET_URL).text
sun_df = pd.read_csv(StringIO(sun_raw))

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
