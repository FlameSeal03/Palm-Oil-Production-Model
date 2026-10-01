import hashlib
import math

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import shape

from config.settings import GEOJSON_PATH, IMAGE_DATE
from src.search.best_sentinel2_date import get_items, CACHE_DIR


# ============================================================
# Configuration
# ============================================================

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

# Public bucket: no AWS credentials needed. These settings also make
# reading small windows from the cloud files faster.
GDAL_OPTIONS = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
}


# ============================================================
# Load plantation boundary
# ============================================================

plantation_gdf = gpd.read_file(GEOJSON_PATH)

# Select the first plantation polygon
plantation = plantation_gdf.iloc[0]

# Boundary in lat/lon, used for the catalog search
plantation_wgs84 = plantation_gdf.to_crs("EPSG:4326").iloc[0].geometry


# ============================================================
# Find the scene for IMAGE_DATE (search results are cached)
# ============================================================

items = get_items(
    plantation_wgs84.__geo_interface__,
    f"{IMAGE_DATE}T00:00:00Z/{IMAGE_DATE}T23:59:59Z",
)

if not items:
    raise ValueError(
        f"No Sentinel-2 scene found for {IMAGE_DATE}. "
        "Run `python -m src.search.best_sentinel2_date` to see available dates."
    )

# Prefer scenes that fully cover the plantation, then the lowest cloud cover
covering = [i for i in items if shape(i.geometry).contains(plantation_wgs84)]

if not covering:
    print(
        "Warning: no single scene fully covers the plantation. "
        "Parts of it may come back empty (NaN)."
    )
    covering = items

item = min(covering, key=lambda i: i.properties.get("eo:cloud_cover", 100))

print("Using scene:", item.id)


# ============================================================
# Read the bands (clipped to the plantation, cached locally)
# ============================================================

def get_asset(name):
    if name not in item.assets:
        raise KeyError(
            f"Asset '{name}' not found. Available assets: {sorted(item.assets)}"
        )
    return item.assets[name]


def scale_and_offset(asset):
    """Convert stored integers to reflectance (0-1)."""
    bands_info = asset.extra_fields.get("raster:bands", [{}])
    info = bands_info[0] if bands_info else {}
    # Fallback values apply to scenes processed with baseline 04.00 or later
    return info.get("scale", 0.0001), info.get("offset", -0.1)


def read_band(asset, bounds, out_shape):
    """Read one band for the given bounds, resampled to the 10 m grid."""
    scale, offset = scale_and_offset(asset)

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

    # A value of 0 means "no data"
    return np.where(dn == 0, np.nan, dn * scale + offset).astype("float32")


def fetch_band_stack():
    """Return (stack, transform, crs). Uses the local cache when available."""
    CACHE_DIR.mkdir(exist_ok=True)

    bounds_key = ",".join(f"{v:.6f}" for v in plantation_wgs84.bounds)
    key = hashlib.sha256(bounds_key.encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"bands_{item.id}_{key}.npz"

    if cache_file.exists():
        print(f"Using cached bands ({cache_file.name}).")
        data = np.load(cache_file, allow_pickle=False)
        return data["stack"], Affine(*data["transform"]), str(data["crs"])

    print("Reading bands from AWS...")

    with rasterio.Env(**GDAL_OPTIONS):
        # Use the red band (10 m) as the reference grid
        with rasterio.open(get_asset("red").href) as ref:
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

        bands = [
            read_band(get_asset(name), bounds, (height, width))
            for name in BAND_ASSETS
        ]

    stack = np.stack(bands)

    np.savez_compressed(
        cache_file,
        stack=stack,
        transform=np.array(transform)[:6],
        crs=crs.to_string(),
    )

    return stack, transform, crs.to_string()


stack, transform, crs = fetch_band_stack()

height, width = stack.shape[1], stack.shape[2]


# ============================================================
# Plantation geometry in the imagery coordinate system
# ============================================================

plantation_utm = plantation_gdf.to_crs(crs)
plantation_utm_feature = plantation_utm.iloc[0]

min_x, min_y, max_x, max_y = plantation_utm_feature.geometry.bounds
bbox_utm = [min_x, min_y, max_x, max_y]

plantation_geometry_utm = plantation_utm_feature.geometry.__geo_interface__

print("Plantation geometry loaded successfully.")
print("Geometry type:", plantation_utm_feature.geometry.geom_type)


# ============================================================
# Mask pixels outside the plantation
# ============================================================

plantation_mask = geometry_mask(
    [plantation_geometry_utm],
    transform=transform,
    invert=True,
    out_shape=(height, width),
)

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
) = [np.where(plantation_mask, band, np.nan) for band in stack]


print("Sentinel-2 data loaded successfully.")
print("Image date:", IMAGE_DATE)
print("Image size:", red.shape)