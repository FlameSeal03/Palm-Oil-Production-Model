import os
from io import BytesIO
from datetime import datetime

import numpy as np
import requests
import rasterio
from dotenv import load_dotenv
from shapely.geometry import shape
from rasterio.features import geometry_mask

from src.common.sentinel2_data import plantation_geometry_utm


# Reusable vegetation-index functions
from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.gndvi import calculate_gndvi
from src.indices.ndmi import calculate_ndmi
from src.indices.evi import calculate_evi


def get_scene_date(scene_id):
    """
    Extract the acquisition date from a Sentinel-2 scene ID.

    Example:
        S2B_MSIL2A_20260808T024529_N0512_R132_T49MEV_20260808T045442

    Returns:
        2026-08-08
    """

    try:
        date_string = scene_id.split("_")[2][:8]

        scene_date = datetime.strptime(
            date_string,
            "%Y%m%d"
        ).date()

        return scene_date

    except (IndexError, ValueError):
        raise ValueError(
            f"Could not extract acquisition date from scene ID:\n"
            f"{scene_id}"
        )


def process_sentinel2_scene(scene_id):
    """
    Retrieve and process one Sentinel-2 scene.

    The function:
    1. Extracts the acquisition date from the scene ID.
    2. Authenticates with Copernicus Data Space.
    3. Loads the plantation geometry in UTM coordinates.
    4. Creates a bounding box around the plantation.
    5. Calculates the required image dimensions at 10 m resolution.
    6. Retrieves Sentinel-2 spectral bands and SCL.
    7. Applies an SCL quality mask.
    8. Applies the exact plantation boundary mask.
    9. Calculates vegetation indices.
    10. Returns the index arrays and quality information.
    """

    print("\nProcessing scene:")
    print(scene_id)

    # ---------------------------------------------------------
    # 1. Determine scene acquisition date
    # ---------------------------------------------------------

    scene_date = get_scene_date(scene_id)

    print("Scene date:", scene_date)

    date_from = f"{scene_date}T00:00:00Z"
    date_to = f"{scene_date}T23:59:59Z"

    # ---------------------------------------------------------
    # 2. Load Sentinel Hub credentials
    # ---------------------------------------------------------

    load_dotenv()

    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Sentinel Hub credentials were not found in .env"
        )

    # ---------------------------------------------------------
    # 3. Get OAuth access token
    # ---------------------------------------------------------

    token_url = (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/protocol/openid-connect/token"
    )

    token_response = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )

    token_response.raise_for_status()

    access_token = token_response.json()["access_token"]

    print("Sentinel Hub authentication successful!")

    # ---------------------------------------------------------
    # 4. Load plantation geometry
    # ---------------------------------------------------------

    plantation_utm = shape(plantation_geometry_utm)

    min_x, min_y, max_x, max_y = plantation_utm.bounds

    bbox_utm = (
        min_x,
        min_y,
        max_x,
        max_y,
    )

    print("\nUTM bounding box:")
    print(bbox_utm)

    # ---------------------------------------------------------
    # 5. Calculate image dimensions
    # ---------------------------------------------------------

    resolution = 10

    width = int(
        np.ceil(
            (max_x - min_x) / resolution
        )
    )

    height = int(
        np.ceil(
            (max_y - min_y) / resolution
        )
    )

    print("\nImage dimensions:")
    print("Width:", width)
    print("Height:", height)
    print("Resolution:", resolution, "meters")

    # ---------------------------------------------------------
    # 6. Sentinel Hub Process API request
    # ---------------------------------------------------------

    process_url = (
        "https://sh.dataspace.copernicus.eu/process/v1"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [{
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
                    "B12",
                    "SCL"
                ],
                units: [
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "DN"
                ]
            }],
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

    request_body = {
        "input": {
            "bounds": {
                "bbox": bbox_utm,
                "properties": {
                    "crs": (
                        "http://www.opengis.net/def/crs/"
                        "EPSG/0/32749"
                    )
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": date_from,
                            "to": date_to,
                        },
                        "mosaickingOrder": "mostRecent",
                    },
                    "processing": {},
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": evalscript,
    }

    # ---------------------------------------------------------
    # 7. Retrieve imagery
    # ---------------------------------------------------------

    response = requests.post(
        process_url,
        headers=headers,
        json=request_body,
        timeout=120,
    )

    if not response.ok:
        print("\nSentinel Hub request failed!")
        print("Status code:", response.status_code)
        print("Response:")
        print(response.text)

    response.raise_for_status()

    print("\nSentinel Hub request successful!")
    print("Status code:", response.status_code)

    # ---------------------------------------------------------
    # 8. Read returned TIFF
    # ---------------------------------------------------------

    with rasterio.open(
        BytesIO(response.content)
    ) as dataset:

        data = dataset.read()

        print("Image shape:", data.shape)
        print("Image dtype:", data.dtype)

        transform = dataset.transform

        dataset_height = dataset.height
        dataset_width = dataset.width

    # ---------------------------------------------------------
    # 9. Separate spectral bands
    # ---------------------------------------------------------

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

    scl = data[10]

    # ---------------------------------------------------------
    # 10. Create SCL quality mask
    # ---------------------------------------------------------

    scl_valid_mask = np.isin(
        scl,
        [4, 5]
    )

    print("\nSCL mask created successfully!")

    total_pixels = scl_valid_mask.size

    scl_usable_pixels = np.sum(
        scl_valid_mask
    )

    scl_masked_pixels = (
        total_pixels -
        scl_usable_pixels
    )

    scl_usable_percentage = (
        scl_usable_pixels /
        total_pixels *
        100
    )

    print(
        "Total pixels:",
        total_pixels
    )

    print(
        "SCL usable pixels:",
        scl_usable_pixels
    )

    print(
        "SCL masked pixels:",
        scl_masked_pixels
    )

    print(
        "SCL usable percentage:",
        scl_usable_percentage
    )

    # ---------------------------------------------------------
    # 11. Create plantation boundary mask
    # ---------------------------------------------------------

    plantation_mask = geometry_mask(
        [plantation_utm.__geo_interface__],
        transform=transform,
        invert=True,
        out_shape=(
            dataset_height,
            dataset_width
        )
    )

    print(
        "\nPlantation boundary mask "
        "created successfully!"
    )

    plantation_pixels = np.sum(
        plantation_mask
    )

    print(
        "Pixels inside plantation:",
        plantation_pixels
    )

    # ---------------------------------------------------------
    # 12. Combine quality masks
    # ---------------------------------------------------------

    valid_mask = (
        scl_valid_mask &
        plantation_mask
    )

    valid_pixels = np.sum(
        valid_mask
    )

    invalid_pixels = (
        total_pixels -
        valid_pixels
    )

    valid_percentage = (
        valid_pixels /
        total_pixels *
        100
    )

    print(
        "\nCombined mask created successfully!"
    )

    print(
        "Valid plantation pixels:",
        valid_pixels
    )

    print(
        "Invalid pixels:",
        invalid_pixels
    )

    print(
        "Valid percentage:",
        valid_percentage
    )

    # ---------------------------------------------------------
    # 13. Apply combined mask to spectral bands
    # ---------------------------------------------------------

    blue = np.where(
        valid_mask,
        blue,
        np.nan
    )

    green = np.where(
        valid_mask,
        green,
        np.nan
    )

    red = np.where(
        valid_mask,
        red,
        np.nan
    )

    red_edge_1 = np.where(
        valid_mask,
        red_edge_1,
        np.nan
    )

    red_edge_2 = np.where(
        valid_mask,
        red_edge_2,
        np.nan
    )

    red_edge_3 = np.where(
        valid_mask,
        red_edge_3,
        np.nan
    )

    nir = np.where(
        valid_mask,
        nir,
        np.nan
    )

    nir_2 = np.where(
        valid_mask,
        nir_2,
        np.nan
    )

    swir_1 = np.where(
        valid_mask,
        swir_1,
        np.nan
    )

    swir_2 = np.where(
        valid_mask,
        swir_2,
        np.nan
    )

    print(
        "\nCombined mask applied "
        "to spectral bands!"
    )

    # ---------------------------------------------------------
    # 14. Calculate vegetation indices
    # ---------------------------------------------------------

    ndvi = calculate_ndvi(
        red,
        nir
    )

    ndre = calculate_ndre(
        red_edge_1,
        nir_2
    )

    gndvi = calculate_gndvi(
        green,
        nir
    )

    ndmi = calculate_ndmi(
        nir,
        swir_1
    )

    evi = calculate_evi(
        blue,
        red,
        nir
    )

    print(
        "\nVegetation indices "
        "calculated successfully!"
    )

    # ---------------------------------------------------------
    # 15. Return results
    # ---------------------------------------------------------

    return {
        "scene_id": scene_id,
        "scene_date": str(scene_date),
        "ndvi": ndvi,
        "ndre": ndre,
        "gndvi": gndvi,
        "ndmi": ndmi,
        "evi": evi,
        "valid_mask": valid_mask,
        "usable_percentage": valid_percentage,
        "transform": transform,
        "width": dataset_width,
        "height": dataset_height,
    }


# -------------------------------------------------------------
# Testing section
# -------------------------------------------------------------

if __name__ == "__main__":

    scene_id = (
        "S2B_MSIL2A_20260815T024529_N0512_R132_T49MEV_"
        "20260815T045442"
    )

    results = process_sentinel2_scene(
        scene_id
    )

    print("\nFinal results:")

    print(
        "Scene:",
        results["scene_id"]
    )

    print(
        "Scene date:",
        results["scene_date"]
    )

    print(
        "Valid plantation pixels:",
        np.sum(
            results["valid_mask"]
        )
    )

    print(
        "Valid percentage:",
        results["usable_percentage"]
    )

    print("\nImage dimensions:")

    print(
        "Width:",
        results["width"]
    )

    print(
        "Height:",
        results["height"]
    )

    print("\nMean vegetation indices:")

    print(
        "NDVI:",
        np.nanmean(
            results["ndvi"]
        )
    )

    print(
        "NDRE:",
        np.nanmean(
            results["ndre"]
        )
    )

    print(
        "GNDVI:",
        np.nanmean(
            results["gndvi"]
        )
    )

    print(
        "NDMI:",
        np.nanmean(
            results["ndmi"]
        )
    )

    print(
        "EVI:",
        np.nanmean(
            results["evi"]
        )
    )
