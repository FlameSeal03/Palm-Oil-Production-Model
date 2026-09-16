import geopandas as gpd
from pystac_client import Client

from config.settings import GEOJSON_PATH, START_DATE, END_DATE


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
    datetime=f"{START_DATE}/{END_DATE}",
)

items = list(search.items())


print(f"Found {len(items)} Sentinel-2 images.")
print(f"Date range: {START_DATE} to {END_DATE}")


# Group images by month
monthly_images = {}

for item in items:
    month = item.datetime.strftime("%Y-%m")

    if month not in monthly_images:
        monthly_images[month] = []

    monthly_images[month].append(item)


# Display results
print("\nImages by month:")

for month in sorted(monthly_images):

    print(f"\n{month}")

    # Sort images by cloud cover
    monthly_images[month].sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover", float("inf")
        )
    )

    for item in monthly_images[month]:

        date = item.datetime.strftime("%Y-%m-%d")
        cloud_cover = item.properties.get("eo:cloud_cover")

        print(
            f"  {date} | "
            f"Cloud cover: {cloud_cover:.2f}% | "
            f"ID: {item.id}"
        )