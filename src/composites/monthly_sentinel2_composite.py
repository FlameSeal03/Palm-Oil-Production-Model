from config.settings import GEOJSON_PATH, START_DATE, END_DATE

import geopandas as gpd
from pystac_client import Client
from collections import defaultdict

MAX_CLOUD_COVER = 80

# Connect to Copernicus Data Space
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
catalog = Client.open(STAC_URL)


# Load plantation boundary
plantation_gdf = gpd.read_file(GEOJSON_PATH)

plantation = plantation_gdf.iloc[0]
plantation_geometry = plantation.geometry.__geo_interface__

print("Plantation geometry loaded successfully.")
print("Geometry type:", plantation.geometry.geom_type)


# Search for Sentinel-2 imagery
search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=plantation_geometry,
    datetime=f"{START_DATE}/{END_DATE}",
    max_items=100,
)

items = list(search.items())

print(f"\nFound {len(items)} Sentinel-2 scenes.")


# Group scenes by month
monthly_scenes = defaultdict(list)

for item in items:
    date = item.datetime
    month = date.strftime("%Y-%m")
    cloud_cover = item.properties.get("eo:cloud_cover")

    if cloud_cover is not None and cloud_cover <= MAX_CLOUD_COVER:
        monthly_scenes[month].append({
            "date": date,
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "id": item.id,
        })


# Sort each month by cloud cover
for month in monthly_scenes:
    monthly_scenes[month].sort(
        key=lambda scene: scene["cloud_cover"]
        if scene["cloud_cover"] is not None
        else 999
    )


# Display results
print("\nMonthly Sentinel-2 scenes:")

for month in sorted(monthly_scenes):
    print(f"\n{month}")

    for scene in monthly_scenes[month]:
        print(
            f"  {scene['date']} | "
            f"Cloud: {scene['cloud_cover']}% | "
            f"{scene['id']}"
        )


# ---------------------------------------------------------
# Test retrieving one Sentinel-2 scene with SCL
# ---------------------------------------------------------

import os
import requests
import numpy as np
import rasterio
from io import BytesIO
from dotenv import load_dotenv
from shapely.geometry import shape
from rasterio.warp import transform_bounds


# Load Sentinel Hub credentials
load_dotenv()

CLIENT_ID = os.getenv("SENTINELHUB_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINELHUB_CLIENT_SECRET")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"


# Get OAuth token
token_response = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
)

token_response.raise_for_status()

access_token = token_response.json()["access_token"]


# Use the clearest August scene for testing
test_scene = "S2B_MSIL2A_20260808T024529_N0512_R132_T49MEV_20260808T045442"

print("\nTesting scene:")
print(test_scene)


# Get plantation bounds in UTM Zone 49S
plantation_shape = shape(plantation_geometry)

bbox_wgs84 = plantation_shape.bounds

bbox_utm = transform_bounds(
    "EPSG:4326",
    "EPSG:32749",
    *bbox_wgs84
)

print("\nUTM bounding box:")
print(bbox_utm)


# Sentinel Hub evalscript
evalscript = """
//VERSION=3

function setup() {
    return {
        input: [
            {
                bands: [
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B11",
                    "B12"
                ],
                units: "REFLECTANCE"
            },
            {
                bands: ["SCL"],
                units: "DN"
            }
        ],
        output: {
            bands: 11,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B05,
        sample.B06,
        sample.B07,
        sample.B08,
        sample.B8A,
        sample.B11,
        sample.B12,
        sample.SCL
    ];
}
"""


# Process API request
request_body = {
    "input": {
        "bounds": {
            "bbox": list(bbox_utm),
            "properties": {
                "crs": "http://www.opengis.net/def/crs/EPSG/0/32749"
            }
        },
        "data": [
            {
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2026-08-08T00:00:00Z",
                        "to": "2026-08-09T00:00:00Z"
                    },
                }
            }
        ]
    },
    "output": {
        "resx": 10,
        "resy": 10,
        "responses": [
            {
                "identifier": "default",
                "format": {
                    "type": "image/tiff"
                }
            }
        ]
    },
    "evalscript": evalscript
}


# Send request
response = requests.post(
    PROCESS_URL,
    headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    },
    json=request_body,
)


# Print the API response if there is an error
if response.status_code != 200:
    print("\nSentinel Hub request failed!")
    print("Status code:", response.status_code)
    print("Response:")
    print(response.text)

response.raise_for_status()

print("\nSentinel Hub request successful!")
print("Status code:", response.status_code)

# Read TIFF
with rasterio.open(BytesIO(response.content)) as dataset:

    data = dataset.read()

    print("Image shape:", data.shape)
    print("Image dtype:", data.dtype)


# Separate SCL band
scl = data[10]

print("\nSCL statistics:")
print("Minimum:", np.nanmin(scl))
print("Maximum:", np.nanmax(scl))