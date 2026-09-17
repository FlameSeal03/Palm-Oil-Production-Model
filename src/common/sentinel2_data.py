import os
import requests
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

from config.settings import GEOJSON_PATH, IMAGE_DATE

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"


# ============================================================
# Load credentials
# ============================================================

load_dotenv()

client_id = os.getenv("SENTINELHUB_CLIENT_ID")
client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")

if not client_id or not client_secret:
    raise ValueError("Sentinel Hub credentials were not found in .env")


# ============================================================
# Authenticate with Copernicus
# ============================================================

response = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    },
)

response.raise_for_status()

token = response.json()["access_token"]


# ============================================================
# Load plantation boundary
# ============================================================

plantation_gdf = gpd.read_file(GEOJSON_PATH)

# Select the first plantation polygon
plantation = plantation_gdf.iloc[0]


# ============================================================
# Reproject to UTM Zone 49S
# ============================================================

plantation_utm = plantation_gdf.to_crs("EPSG:32749")

plantation_utm_feature = plantation_utm.iloc[0]


# ============================================================
# Get bounding box
# ============================================================

min_x, min_y, max_x, max_y = plantation_utm_feature.geometry.bounds

bbox_utm = [
    min_x,
    min_y,
    max_x,
    max_y
]

plantation_geometry_utm = plantation_utm_feature.geometry.__geo_interface__

print("Plantation geometry loaded successfully.")
print("Geometry type:", plantation_utm_feature.geometry.geom_type)

# ============================================================
# Sentinel-2 request
# ============================================================

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
            "B12"
        ],
        output: {
            bands: 10,
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
        sample.B12
    ];
}
"""


request_body = {
    "input": {
        "bounds": {
            "bbox": bbox_utm,
            "properties": {
                "crs": "http://www.opengis.net/def/crs/EPSG/0/32749"
            }
        },
        "data": [
            {
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{IMAGE_DATE}T00:00:00Z",
                        "to": f"{IMAGE_DATE}T23:59:59Z"
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


headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}


response = requests.post(
    PROCESS_URL,
    headers=headers,
    json=request_body
)

response.raise_for_status()


# ============================================================
# Read returned GeoTIFF
# ============================================================

with MemoryFile(response.content) as memfile:

    with memfile.open() as dataset:

        # Create a mask for pixels outside the plantation
        plantation_mask = geometry_mask(
            [plantation_geometry_utm],
            transform=dataset.transform,
            invert=True,
            out_shape=(dataset.height, dataset.width)
        )

        # Apply the plantation mask to every band
        blue = np.where(plantation_mask, blue, np.nan)
        green = np.where(plantation_mask, green, np.nan)
        red = np.where(plantation_mask, red, np.nan)
        red_edge_1 = np.where(plantation_mask, red_edge_1, np.nan)
        red_edge_2 = np.where(plantation_mask, red_edge_2, np.nan)
        red_edge_3 = np.where(plantation_mask, red_edge_3, np.nan)
        nir = np.where(plantation_mask, nir, np.nan)
        nir_2 = np.where(plantation_mask, nir_2, np.nan)
        swir_1 = np.where(plantation_mask, swir_1, np.nan)
        swir_2 = np.where(plantation_mask, swir_2, np.nan)



print("Sentinel-2 data loaded successfully.")
print("Image date:", IMAGE_DATE)
print("Image size:", red.shape)