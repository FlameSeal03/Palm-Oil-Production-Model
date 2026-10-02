"""
Read-only diagnostic: compares what the STAC metadata says about each band's
scale/offset with the raw values actually stored in the AWS files.

Run from the project root (needs the search cache from a previous run):
    python diagnose_scale.py
"""

import glob
import json

import geopandas as gpd
import numpy as np
import rasterio
from pystac import Item
from rasterio.windows import Window

from config.settings import GEOJSON_PATH

SCENE_ID = "S2A_49MHT_20260720_0_L2A"

# Find the scene in the cached search results
item = None
for path in glob.glob("cache/search_*.json"):
    with open(path) as f:
        for d in json.load(f):
            if d["id"] == SCENE_ID:
                item = Item.from_dict(d)

if item is None:
    raise SystemExit(
        f"{SCENE_ID} not found in cache/search_*.json. "
        "Run `python -m src.indices.ndvi` once first so the search is cached."
    )

print("Scene:", item.id)

print("\nScene properties mentioning offset / baseline / processing:")
for key, value in item.properties.items():
    if any(w in key.lower() for w in ("offset", "baseline", "processing", "version")):
        print(f"  {key}: {value}")

plantation = gpd.read_file(GEOJSON_PATH)

with rasterio.Env(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
):
    for name in ("blue", "green", "red", "nir"):
        asset = item.assets[name]

        print(f"\n--- {name} ---")
        print("STAC raster:bands:", asset.extra_fields.get("raster:bands"))

        with rasterio.open(asset.href) as src:
            print("File dtype:", src.dtypes[0], "| nodata:", src.nodata)
            print("File scales / offsets:", src.scales, src.offsets)

            # Sample a 100 x 100 pixel window at the plantation centre
            centre = plantation.to_crs(src.crs).geometry.iloc[0].centroid
            row, col = src.index(centre.x, centre.y)
            half = 50
            window = Window(col - half, row - half, 2 * half, 2 * half)

            dn = src.read(1, window=window, boundless=True, fill_value=0)
            valid = dn[dn > 0]

            if valid.size == 0:
                print("No valid pixels in the sample window.")
                continue

            p5, median, p95 = np.percentile(valid, [5, 50, 95])
            print(f"Raw stored values  p5: {p5:.0f}  median: {median:.0f}  p95: {p95:.0f}")