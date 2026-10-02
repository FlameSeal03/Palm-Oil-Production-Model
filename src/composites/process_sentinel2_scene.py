import hashlib
import json
import math
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import pystac_client
from affine import Affine
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from rasterio.windows import transform as window_transform

from config.settings import GEOJSON_PATH, IMAGE_DATE
from src.search.best_sentinel2_date import STAC_URL, COLLECTION, CACHE_DIR


# Reusable vegetation-index functions
from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.gndvi import calculate_gndvi
from src.indices.ndmi import calculate_ndmi
from src.indices.evi import calculate_evi


# Output order matches the original script:
# B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
BAND_ASSETS = [
    "blue",       # B02
    "green",      # B03
    "red",        # B04
    "rededge1",   # B05
    "rededge2",   # B06
    "rededge3",   # B07
    "nir",        # B08
    "nir08",      # B8A
    "swir16",     # B11
    "swir22",     # B12
]

# Public bucket: no AWS credentials needed.
GDAL_OPTIONS = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}

# Plantation boundary (read once; no network needed)
plantation_gdf = gpd.read_file(GEOJSON_PATH)
plantation_wgs84 = plantation_gdf.to_crs("EPSG:4326").iloc[0].geometry


def get_scene_date(scene_id):
    """
    Extract the acquisition date from a Sentinel-2 scene ID.

    Works for AWS Earth Search IDs:
        S2B_49MEV_20260815_0_L2A

    and for Copernicus-style IDs:
        S2B_MSIL2A_20260815T024529_N0512_R132_T49MEV_20260815T045442

    Returns:
        2026-08-15
    """

    try:
        date_string = scene_id.split("_")[2][:8]

        return datetime.strptime(date_string, "%Y%m%d").date()

    except (IndexError, ValueError):
        raise ValueError(
            f"Could not extract acquisition date from scene ID:\n"
            f"{scene_id}"
        )


def get_item(scene):
    """
    Return a STAC item for a scene.

    `scene` can be a pystac Item (used as is) or an AWS Earth Search
    scene ID such as "S2B_49MEV_20260815_0_L2A". Lookups are cached.
    """

    if isinstance(scene, Item):
        return scene

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"item_{scene}.json"

    if cache_file.exists():
        with cache_file.open() as f:
            return Item.from_dict(json.load(f))

    catalog = Client.open(STAC_URL)
    results = list(
        catalog.search(collections=[COLLECTION], ids=[scene]).items()
    )

    if not results:
        raise ValueError(
            f"Scene '{scene}' was not found in the AWS catalog. "
            "AWS scene IDs look like S2B_49MEV_20260815_0_L2A "
            "(Copernicus product names will not match)."
        )

    with cache_file.open("w") as f:
        json.dump(results[0].to_dict(), f)

    return results[0]


def get_asset(item, name):
    if name not in item.assets:
        raise KeyError(
            f"Asset '{name}' not found. Available assets: {sorted(item.assets)}"
        )
    return item.assets[name]


def scale_and_offset(asset, item):
    """Convert stored integers to reflectance (0-1).

    The STAC metadata lists an offset of -0.1, but when the scene property
    'earthsearch:boa_offset_applied' is true the offset has already been
    applied to the stored values. Applying it again produced negative
    reflectance and impossible index values, so it is skipped in that case.
    """
    bands_info = asset.extra_fields.get("raster:bands") or [{}]
    info = bands_info[0]

    scale = info.get("scale", 0.0001)

    if item.properties.get("earthsearch:boa_offset_applied", False):
        offset = 0.0
    else:
        offset = info.get("offset", 0.0)

    return scale, offset


def read_band(asset, item, bounds, out_shape, reflectance=True):
    """Read one band for the given bounds, resampled to the 10 m grid."""

    with rasterio.open(asset.href) as src:
        window = from_bounds(*bounds, transform=src.transform)
        dn = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=0,
        )

    if not reflectance:
        return dn

    scale, offset = scale_and_offset(asset, item)

    # A value of 0 means "no data"
    return np.where(dn == 0, np.nan, dn * scale + offset).astype("float32")


def fetch_scene_arrays(item):
    """
    Return (bands, scl, transform, crs) clipped to the plantation.

    bands is shaped (10, height, width) in BAND_ASSETS order.
    Results are cached locally so reruns do not download again.
    """

    CACHE_DIR.mkdir(exist_ok=True)

    bounds_key = ",".join(f"{v:.6f}" for v in plantation_wgs84.bounds)
    key = hashlib.sha256(bounds_key.encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"scene_{item.id}_{key}.npz"

    if cache_file.exists():
        print(f"Using cached scene ({cache_file.name}).")
        data = np.load(cache_file, allow_pickle=False)
        return (
            data["bands"],
            data["scl"],
            Affine(*data["transform"]),
            str(data["crs"]),
        )

    with rasterio.Env(**GDAL_OPTIONS):

        # Use the red band (10 m) as the reference grid
        with rasterio.open(get_asset(item, "red").href) as ref:
            crs = ref.crs
            ref_transform = ref.transform

        plantation_in_crs = plantation_gdf.to_crs(crs).iloc[0].geometry
        min_x, min_y, max_x, max_y = plantation_in_crs.bounds

        # Snap the bounding box outward to whole 10 m pixels
        col_min, row_min = ~ref_transform * (min_x, max_y)
        col_max, row_max = ~ref_transform * (max_x, min_y)

        col_off = math.floor(col_min)
        row_off = math.floor(row_min)
        width = math.ceil(col_max) - col_off
        height = math.ceil(row_max) - row_off

        window = Window(col_off, row_off, width, height)
        bounds = window_bounds(window, ref_transform)
        transform = window_transform(window, ref_transform)

        bands = np.stack([
            read_band(get_asset(item, name), item, bounds, (height, width))
            for name in BAND_ASSETS
        ])

        # Scene Classification Layer: categorical, so no scaling
        scl = read_band(
            get_asset(item, "scl"),
            item,
            bounds,
            (height, width),
            reflectance=False,
        ).astype("uint8")

    # Fail loudly rather than cache bad data: red reflectance over a
    # plantation should never have a negative median.
    if np.nanmedian(bands[2]) < 0:
        raise ValueError(
            "Median red reflectance is negative: the scale/offset looks wrong. "
            "Check the 'raster:bands' metadata for this scene."
        )

    np.savez_compressed(
        cache_file,
        bands=bands,
        scl=scl,
        transform=np.array(transform)[:6],
        crs=crs.to_string(),
    )

    return bands, scl, transform, crs.to_string()


def process_sentinel2_scene(scene):
    """
    Retrieve and process one Sentinel-2 scene from AWS.

    `scene` is an AWS scene ID (for example S2B_49MEV_20260815_0_L2A)
    or a pystac Item.

    The function:
    1. Looks up the scene and its acquisition date.
    2. Reads the plantation area of each spectral band and the SCL
       from AWS (or from the local cache).
    3. Applies an SCL quality mask.
    4. Applies the exact plantation boundary mask.
    5. Calculates vegetation indices.
    6. Returns the index arrays and quality information.
    """

    # ---------------------------------------------------------
    # 1. Find the scene and its acquisition date
    # ---------------------------------------------------------

    item = get_item(scene)
    scene_id = item.id

    scene_date = (
        item.datetime.date() if item.datetime else get_scene_date(scene_id)
    )

    print("Scene date:", scene_date)

    # ---------------------------------------------------------
    # 2. Read bands and SCL
    # ---------------------------------------------------------

    bands, scl, transform, crs = fetch_scene_arrays(item)

    dataset_height, dataset_width = scl.shape

    # ---------------------------------------------------------
    # 3. Create SCL quality mask (4 = vegetation, 5 = not vegetated)
    # ---------------------------------------------------------

    scl_valid_mask = np.isin(scl, [4, 5])

    total_pixels = scl_valid_mask.size

    # ---------------------------------------------------------
    # 4. Create plantation boundary mask
    # ---------------------------------------------------------

    plantation_utm = plantation_gdf.to_crs(crs).iloc[0].geometry

    plantation_mask = geometry_mask(
        [plantation_utm.__geo_interface__],
        transform=transform,
        invert=True,
        out_shape=(dataset_height, dataset_width),
    )

    plantation_scl = scl[plantation_mask]

    plantation_scl_classes, plantation_scl_counts = np.unique(
        plantation_scl,
        return_counts=True,
    )

    for scl_class, count in zip(
        plantation_scl_classes,
        plantation_scl_counts,
    ):
        print("  SCL class:", scl_class, "Pixels:", count)

    # ---------------------------------------------------------
    # 5. Combine quality masks
    # ---------------------------------------------------------

    valid_mask = scl_valid_mask & plantation_mask

    valid_pixels = np.sum(valid_mask)

    # As in the original script, this is relative to the full
    # bounding box, not only the pixels inside the plantation.
    valid_percentage = valid_pixels / total_pixels * 100

    # ---------------------------------------------------------
    # 6. Apply combined mask to spectral bands
    # ---------------------------------------------------------

    (
        blue,
        green,
        red,
        red_edge_1,
        red_edge_2,
        red_edge_3,
        nir,
        nir_2,
        swir_1,
        swir_2,
    ) = [np.where(valid_mask, band, np.nan) for band in bands]

    # ---------------------------------------------------------
    # 7. Calculate vegetation indices
    # ---------------------------------------------------------

    ndvi = calculate_ndvi(red, nir)

    ndre = calculate_ndre(red_edge_1, nir_2)

    gndvi = calculate_gndvi(green, nir)

    ndmi = calculate_ndmi(nir, swir_1)

    evi = calculate_evi(blue, red, nir)

    print("\nVegetation indices calculated successfully!")

    # ---------------------------------------------------------
    # 8. Return results
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

    # TODO
    scene_id = "S2B_49MEV_20260815_0_L2A"

    results = process_sentinel2_scene(scene_id)

    print("\nFinal results:")

    print("Scene:", results["scene_id"])
    print("Scene date:", results["scene_date"])

    print(
        "Valid plantation pixels:",
        np.sum(results["valid_mask"]),
    )

    print("Valid percentage:", results["usable_percentage"])

    print("\nImage dimensions:")
    print("Width:", results["width"])
    print("Height:", results["height"])

    print("\nMean vegetation indices:")
    print("NDVI:", np.nanmean(results["ndvi"]))
    print("NDRE:", np.nanmean(results["ndre"]))
    print("GNDVI:", np.nanmean(results["gndvi"]))
    print("NDMI:", np.nanmean(results["ndmi"]))
    print("EVI:", np.nanmean(results["evi"]))