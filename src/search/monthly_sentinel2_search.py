import argparse

import geopandas as gpd

from config.settings import GEOJSON_PATH, START_DATE, END_DATE
from src.search.best_sentinel2_date import get_items


def get_monthly_items(refresh=False):
    """
    Return Sentinel-2 images for the plantation grouped by month.

    Result: {"YYYY-MM": [items sorted by cloud cover, lowest first]}

    Uses the same cached search as best_sentinel2_date.py, so if that
    script has already run for this plantation and date range, no new
    request is made.
    """

    # Load plantation boundary
    plantation_gdf = gpd.read_file(GEOJSON_PATH)

    # Use the first feature for now
    plantation = plantation_gdf.iloc[0]

    # Get the plantation geometry
    plantation_geometry = plantation.geometry.__geo_interface__

    items = get_items(
        plantation_geometry,
        f"{START_DATE}/{END_DATE}",
        refresh=refresh,
    )

    # Group images by month
    monthly_images = {}

    for item in items:
        month = item.datetime.strftime("%Y-%m")
        monthly_images.setdefault(month, []).append(item)

    # Sort each month's images by cloud cover
    for month_items in monthly_images.values():
        month_items.sort(
            key=lambda item: item.properties.get(
                "eo:cloud_cover", float("inf")
            )
        )

    return monthly_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cache and run a new search.",
    )
    args = parser.parse_args()

    monthly_images = get_monthly_items(refresh=args.refresh)

    total = sum(len(v) for v in monthly_images.values())

    print(f"Found {total} Sentinel-2 images.")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Display results
    print("\nImages by month:")

    for month in sorted(monthly_images):

        print(f"\n{month}")

        for item in monthly_images[month]:

            date = item.datetime.strftime("%Y-%m-%d")
            cloud_cover = item.properties.get("eo:cloud_cover")

            cloud_text = (
                f"{cloud_cover:.2f}%" if cloud_cover is not None else "n/a"
            )

            print(
                f"  {date} | "
                f"Cloud cover: {cloud_text} | "
                f"ID: {item.id}"
            )


if __name__ == "__main__":
    main()