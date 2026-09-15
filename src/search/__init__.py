import geopandas as gpd
from pystac_client import Client

from config.settings import GEOJSON_PATH


# Connect to Copernicus Data Space STAC
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
catalog = Client.open(STAC_URL)


# Load plantation boundary
plantation_gdf = gpd.read_file(GEOJSON_PATH)

# Use the first feature for now
plantation = plantation_gdf.iloc[0]

# Get the plantation geometry
plantation_geometry = plantation.geometry.__geo_interface__


# Search for Sentinel-2 Level-2A imagery
search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=plantation_geometry,
    datetime="2026-01-01/2026-12-31",
)

items = list(search.items())


print(f"Found {len(items)} Sentinel-2 images.")

if len(items) == 0:
    print("No Sentinel-2 images found.")
else:
    # Sort images by cloud cover
    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover", float("inf")
        )
    )

    print("\nAvailable images:")
    
    for item in items[:10]:
        date = item.datetime.strftime("%Y-%m-%d")
        cloud_cover = item.properties.get("eo:cloud_cover")

        print(
            f"{date} | Cloud cover: {cloud_cover:.2f}% | "
            f"ID: {item.id}"
        )

    # Best image based on scene-level cloud cover
    best_item = items[0]

    best_date = best_item.datetime.strftime("%Y-%m-%d")
    best_cloud_cover = best_item.properties.get("eo:cloud_cover")

    print("\nBest Sentinel-2 image:")
    print(f"Date: {best_date}")
    print(f"Cloud cover: {best_cloud_cover:.2f}%")
    print(f"Image ID: {best_item.id}")