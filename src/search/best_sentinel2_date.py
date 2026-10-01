import argparse
import hashlib
import json
import time
from pathlib import Path

import geopandas as gpd
from pystac import Item
from pystac_client import Client

from config.settings import GEOJSON_PATH, START_DATE, END_DATE


# AWS-hosted Sentinel-2 L2A catalog (Earth Search by Element 84). No credentials needed.
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Search results are cached here so repeated runs do not hit the network.
CACHE_DIR = Path("cache")
# New scenes keep arriving while the date range is still open, so cached
# results expire. Use --refresh to force a new search at any time.
CACHE_MAX_AGE_HOURS = 24


def make_cache_path(geometry, datetime_range):
    """Build a cache filename from everything that defines the search."""
    key_source = json.dumps(
        {
            "stac_url": STAC_URL,
            "collection": COLLECTION,
            "geometry": geometry,
            "datetime": datetime_range,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(key_source.encode()).hexdigest()[:16]
    return CACHE_DIR / f"search_{key}.json"


def load_cached_items(path):
    """Return cached items if the cache exists and is fresh, otherwise None."""
    if not path.exists():
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > CACHE_MAX_AGE_HOURS:
        return None

    try:
        with path.open() as f:
            return [Item.from_dict(d) for d in json.load(f)]
    except (json.JSONDecodeError, KeyError, ValueError):
        # Corrupt cache file: ignore it and search again.
        return None


def save_cached_items(path, items):
    CACHE_DIR.mkdir(exist_ok=True)
    with path.open("w") as f:
        json.dump([item.to_dict() for item in items], f)


def get_items(geometry, datetime_range, refresh=False):
    """Get Sentinel-2 items from the cache, or from the STAC API if needed."""
    cache_path = make_cache_path(geometry, datetime_range)

    if not refresh:
        cached = load_cached_items(cache_path)
        if cached is not None:
            print(f"Using cached search results ({cache_path.name}).")
            return cached

    print("Searching the STAC catalog...")
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        intersects=geometry,
        datetime=datetime_range,
    )
    items = list(search.items())

    # Do not cache an empty result, which may come from a temporary problem.
    if items:
        save_cached_items(cache_path, items)

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cache and run a new search.",
    )
    args = parser.parse_args()

    # Load plantation boundary
    plantation_gdf = gpd.read_file(GEOJSON_PATH)

    # Use the first feature for now
    plantation = plantation_gdf.iloc[0]
    plantation_geometry = plantation.geometry.__geo_interface__

    items = get_items(
        plantation_geometry,
        f"{START_DATE}/{END_DATE}",
        refresh=args.refresh,
    )

    print(f"Found {len(items)} Sentinel-2 images.")

    # Skip scenes with no cloud-cover value so formatting does not fail
    items = [
        item for item in items
        if item.properties.get("eo:cloud_cover") is not None
    ]

    if len(items) == 0:
        print("No Sentinel-2 images found.")
        return

    # Sort images by cloud cover
    items.sort(key=lambda item: item.properties["eo:cloud_cover"])

    print("\nAvailable images:")
    for item in items[:10]:
        date = item.datetime.strftime("%Y-%m-%d")
        cloud_cover = item.properties["eo:cloud_cover"]
        print(f"{date} | Cloud cover: {cloud_cover:.2f}% | ID: {item.id}")

    # Best image based on scene-level cloud cover
    best_item = items[0]
    print("\nBest Sentinel-2 image:")
    print(f"Date: {best_item.datetime.strftime('%Y-%m-%d')}")
    print(f"Cloud cover: {best_item.properties['eo:cloud_cover']:.2f}%")
    print(f"Image ID: {best_item.id}")


if __name__ == "__main__":
    main()