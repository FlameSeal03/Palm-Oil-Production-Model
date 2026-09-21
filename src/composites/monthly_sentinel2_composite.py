from config.settings import GEOJSON_PATH, START_DATE, END_DATE

import geopandas as gpd
from pystac_client import Client
from collections import defaultdict
from datetime import datetime, timedelta
import time

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


# Search for Sentinel-2 imagery month by month
monthly_scenes = defaultdict(list)

start = datetime.fromisoformat(START_DATE)
end = datetime.fromisoformat(END_DATE)

current = start.replace(day=1)

total_scenes = 0

while current <= end:

    # First day of current month
    month_start = current

    # First day of next month
    if current.month == 12:
        next_month = current.replace(
            year=current.year + 1,
            month=1
        )
    else:
        next_month = current.replace(
            month=current.month + 1
        )

    month_end = next_month - timedelta(seconds=1)

    # Do not search past END_DATE
    if month_end > end:
        month_end = end

    month_name = current.strftime("%Y-%m")

    print(f"\nSearching {month_name}...")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=plantation_geometry,
        datetime=(
            f"{month_start.isoformat()}Z/"
            f"{month_end.isoformat()}Z"
        ),
        max_items=100,
    )

    month_items = list(search.items())

    time.sleep(3)

    print(f"Found {len(month_items)} scenes.")

    total_scenes += len(month_items)

    # Store scenes that pass the cloud threshold
    for item in month_items:

        cloud_cover = item.properties.get("eo:cloud_cover")

        if cloud_cover is not None and cloud_cover <= MAX_CLOUD_COVER:

            monthly_scenes[month_name].append({
                "date": item.datetime,
                "cloud_cover": cloud_cover,
                "id": item.id,
            })

    current = next_month


print(f"\nTotal scenes found: {total_scenes}")


# Sort each month by cloud cover
for month in monthly_scenes:

    monthly_scenes[month].sort(
        key=lambda x: x["cloud_cover"]
    )

    print(f"\n{month}:")

    for item in monthly_scenes[month]:

        print(
            item["date"],
            "| Cloud:",
            item["cloud_cover"],
            "|",
            item["id"]
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
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
            "SCL"
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
                    }
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


# Separate bands
blue = data[0]
green = data[1]
red = data[2]
red_edge_1 = data[3]
red_edge_2 = data[4]
red_edge_3 = data[5]
nir = data[6]
nir_2 = data[7]
swir_1 = data[8]
swir_2 = data[9]

# Separate SCL band
scl = data[10]

print("\nSCL statistics:")
print("Minimum:", np.nanmin(scl))
print("Maximum:", np.nanmax(scl))

# Create a mask for usable pixels
# SCL classes 4 and 5 are vegetation and not-vegetated.
# We remove cloud shadows, clouds, cirrus, and other invalid classes.

valid_mask = np.isin(
    scl,
    [4, 5]
)

# Apply the SCL mask to all spectral bands
blue = np.where(valid_mask, blue, np.nan)
green = np.where(valid_mask, green, np.nan)
red = np.where(valid_mask, red, np.nan)
red_edge_1 = np.where(valid_mask, red_edge_1, np.nan)
red_edge_2 = np.where(valid_mask, red_edge_2, np.nan)
red_edge_3 = np.where(valid_mask, red_edge_3, np.nan)
nir = np.where(valid_mask, nir, np.nan)
nir_2 = np.where(valid_mask, nir_2, np.nan)
swir_1 = np.where(valid_mask, swir_1, np.nan)
swir_2 = np.where(valid_mask, swir_2, np.nan)

print("\nSCL mask created successfully!")

print("\nSCL mask applied to spectral bands!")

print("Total pixels:", valid_mask.size)
print("Usable pixels:", np.sum(valid_mask))
print("Masked pixels:", np.sum(~valid_mask))
print(
    "Usable percentage:",
    np.sum(valid_mask) / valid_mask.size * 100
)

# Calculate vegetation indices using the masked bands

# NDVI
ndvi = np.where(
    (nir + red) != 0,
    (nir - red) / (nir + red),
    np.nan
)

# NDRE
ndre = np.where(
    (nir_2 + red_edge_1) != 0,
    (nir_2 - red_edge_1) / (nir_2 + red_edge_1),
    np.nan
)

# GNDVI
gndvi = np.where(
    (nir + green) != 0,
    (nir - green) / (nir + green),
    np.nan
)

# NDMI
ndmi = np.where(
    (nir + swir_1) != 0,
    (nir - swir_1) / (nir + swir_1),
    np.nan
)

# EVI
evi_denominator = nir + 6 * red - 7.5 * blue + 1

evi = np.where(
    evi_denominator != 0,
    2.5 * (nir - red) / evi_denominator,
    np.nan
)

print("\nVegetation indices calculated successfully!")

print("\nAugust 2026 test scene statistics:")

print("NDVI mean:", np.nanmean(ndvi))
print("NDRE mean:", np.nanmean(ndre))
print("GNDVI mean:", np.nanmean(gndvi))
print("NDMI mean:", np.nanmean(ndmi))
print("EVI mean:", np.nanmean(evi))