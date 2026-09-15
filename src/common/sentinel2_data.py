import os
import requests
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

GEOJSON_PATH = "data/boundaries/PT.geojson"

IMAGE_DATE = "2026-08-08"

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

        blue = dataset.read(1)
        green = dataset.read(2)
        red = dataset.read(3)
        red_edge_1 = dataset.read(4)
        red_edge_2 = dataset.read(5)
        red_edge_3 = dataset.read(6)
        nir = dataset.read(7)
        nir_2 = dataset.read(8)
        swir_1 = dataset.read(9)
        swir_2 = dataset.read(10)


print("Sentinel-2 data loaded successfully.")
print("Image date:", IMAGE_DATE)
print("Image size:", red.shape)