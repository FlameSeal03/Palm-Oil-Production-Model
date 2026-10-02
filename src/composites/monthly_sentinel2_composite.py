import argparse
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np

from config.settings import (
    GEOJSON_PATH,
    START_DATE,
    END_DATE,
)

from src.search.best_sentinel2_date import get_items
from src.composites.process_sentinel2_scene import (
    process_sentinel2_scene,
)


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

CLOUD_THRESHOLD = 80.0

# Indices combined into monthly composites: (display name, results key)
INDICES = [
    ("NDVI", "ndvi"),
    ("NDRE", "ndre"),
    ("GNDVI", "gndvi"),
    ("NDMI", "ndmi"),
    ("EVI", "evi"),
]


# -------------------------------------------------------------
# Select usable scenes for one month
# -------------------------------------------------------------

def select_monthly_scenes(month_items, month_name):
    """
    Choose the usable Sentinel-2 scenes for one month.

    Only scenes with cloud cover at or below CLOUD_THRESHOLD
    are retained, sorted by acquisition date.
    """

    print(
        f"\nScenes found in {month_name}:",
        len(month_items),
    )

    # ---------------------------------------------------------
    # Filter by cloud cover
    # ---------------------------------------------------------

    usable_scenes = []

    for scene in month_items:

        cloud_cover = scene.properties.get(
            "eo:cloud_cover"
        )

        if cloud_cover is None:
            continue

        if cloud_cover <= CLOUD_THRESHOLD:
            usable_scenes.append(scene)

    # ---------------------------------------------------------
    # Sort by acquisition date
    # ---------------------------------------------------------

    usable_scenes.sort(
        key=lambda scene: scene.datetime
    )

    print(
        "Scenes retained after cloud filtering:",
        len(usable_scenes),
    )

    for scene in usable_scenes:

        cloud_cover = scene.properties.get(
            "eo:cloud_cover"
        )

        print(
            "  ",
            scene.datetime.date(),
            f"{cloud_cover:.2f}%",
            scene.id,
        )

    return usable_scenes


# -------------------------------------------------------------
# Create monthly composite
# -------------------------------------------------------------

def create_monthly_composite(
    scenes,
    month_name,
):
    """
    Process all Sentinel-2 scenes for a month
    and create pixel-wise median composites.

    Returns:
        Dictionary containing:
            - monthly NDVI
            - monthly NDRE
            - monthly GNDVI
            - monthly NDMI
            - monthly EVI
            - observation count
    """

    if not scenes:
        print(
            f"\nNo usable scenes found for {month_name}."
        )

        return None

    print(
        f"\nCreating composite for {month_name}"
    )

    # One list of arrays (one layer per scene) for each index
    arrays = {key: [] for _, key in INDICES}

    # Pixel grid of the first processed scene, used to make sure
    # every later scene lines up exactly with it
    reference_grid = None

    # ---------------------------------------------------------
    # Process every scene
    # ---------------------------------------------------------

    for index, scene in enumerate(
        scenes,
        start=1,
    ):

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Processing scene "
            f"{index}/{len(scenes)}"
        )

        print(
            f"Date: {scene.datetime.date()}"
        )

        print(
            f"Cloud cover: "
            f"{scene.properties.get('eo:cloud_cover'):.2f}%"
        )

        print(
            f"Scene ID: {scene.id}"
        )

        # -----------------------------------------------------
        # Process individual scene (the item is passed directly,
        # so no extra catalog lookup is needed)
        # -----------------------------------------------------

        results = process_sentinel2_scene(
            scene
        )

        # -----------------------------------------------------
        # Make sure this scene's pixels line up with the others
        # -----------------------------------------------------

        grid = (
            results["transform"],
            results["height"],
            results["width"],
        )

        if reference_grid is None:
            reference_grid = grid

        elif grid != reference_grid:
            print(
                "  Skipping this scene: its pixel grid does not match "
                "the other scenes this month (possibly a different "
                "coordinate system)."
            )
            continue

        for _, key in INDICES:
            arrays[key].append(results[key])

    if not arrays["ndvi"]:
        print(
            f"\nNo scenes could be combined for {month_name}."
        )

        return None

    # ---------------------------------------------------------
    # Stack scene arrays
    # ---------------------------------------------------------

    stacks = {
        key: np.stack(layers, axis=0)
        for key, layers in arrays.items()
    }

    # ---------------------------------------------------------
    # Calculate number of valid observations
    # ---------------------------------------------------------

    observation_count = np.sum(
        ~np.isnan(stacks["ndvi"]),
        axis=0,
    )

    # ---------------------------------------------------------
    # Calculate pixel-wise monthly medians
    # ---------------------------------------------------------

    composite = {"month": month_name}

    for _, key in INDICES:
        composite[key] = np.nanmedian(
            stacks[key],
            axis=0,
        )

    composite["observation_count"] = observation_count

    return composite


# -------------------------------------------------------------
# Print composite statistics
# -------------------------------------------------------------

def print_composite_statistics(
    composite,
):
    """
    Print summary statistics for a monthly composite.
    """

    print(
        "\n========================================"
    )

    print(
        f"Monthly Composite: "
        f"{composite['month']}"
    )

    print(
        "========================================"
    )

    for name, key in INDICES:

        print(f"\n{name}:")
        print(
            "  Mean:",
            np.nanmean(composite[key]),
        )
        print(
            "  Median:",
            np.nanmedian(composite[key]),
        )
        print(
            "  Standard deviation:",
            np.nanstd(composite[key]),
        )

    print(
        " Number of scenes:",
        np.nanmax(composite["observation_count"]),
    )


class Tee:
    """
    Write output to both the terminal and a log file.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


# -------------------------------------------------------------
# Main program
# -------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cached scene search and run a new one.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Set up log file
    # ---------------------------------------------------------

    log_directory = Path("logs")

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        log_directory /
        "monthly_sentinel2_composite.log"
    )

    # ---------------------------------------------------------
    # Open log file (overwritten on each run)
    # ---------------------------------------------------------

    with open(
        log_file,
        "w",
        encoding="utf-8",
    ) as log:

        original_stdout = sys.stdout

        sys.stdout = Tee(
            original_stdout,
            log,
        )

        try:

            # -------------------------------------------------
            # Start of run
            # -------------------------------------------------

            print(
                "\n\n========================================"
            )

            print(
                "Monthly Sentinel-2 Composite Run"
            )

            print(
                "========================================"
            )

            print(
                "Run started:",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )

            print("Plantation:", GEOJSON_PATH)

            # -------------------------------------------------
            # Load plantation geometry
            # -------------------------------------------------

            plantation_gdf = gpd.read_file(GEOJSON_PATH)

            plantation_geometry = plantation_gdf.iloc[0].geometry

            print(
                "Plantation geometry loaded."
            )

            print(
                "Geometry type:",
                plantation_geometry.geom_type,
            )

            # -------------------------------------------------
            # Search for scenes once for the whole date range
            # (cached, and shared with the other search scripts)
            # -------------------------------------------------

            all_items = get_items(
                plantation_geometry.__geo_interface__,
                f"{START_DATE}/{END_DATE}",
                refresh=args.refresh,
            )

            print(
                f"Scenes found for {START_DATE} to {END_DATE}:",
                len(all_items),
            )

            # Group scenes by the month they were acquired in
            items_by_month = {}

            for item in all_items:
                month_key = item.datetime.strftime("%Y-%m")
                items_by_month.setdefault(month_key, []).append(item)

            # -------------------------------------------------
            # Create monthly date ranges
            # -------------------------------------------------

            start_year = int(
                START_DATE[:4]
            )

            start_month = int(
                START_DATE[5:7]
            )

            end_year = int(
                END_DATE[:4]
            )

            end_month = int(
                END_DATE[5:7]
            )

            current_year = start_year
            current_month = start_month

            # -------------------------------------------------
            # Process each month
            # -------------------------------------------------

            while (
                current_year < end_year
                or (
                    current_year == end_year
                    and current_month <= end_month
                )
            ):

                month_name = (
                    f"{current_year:04d}-"
                    f"{current_month:02d}"
                )

                # ---------------------------------------------
                # Select scenes for this month
                # ---------------------------------------------

                scenes = select_monthly_scenes(
                    items_by_month.get(month_name, []),
                    month_name,
                )

                # ---------------------------------------------
                # Create composite
                # ---------------------------------------------

                composite = create_monthly_composite(
                    scenes,
                    month_name,
                )

                # ---------------------------------------------
                # Print results
                # ---------------------------------------------

                if composite is not None:

                    print_composite_statistics(
                        composite
                    )

                # ---------------------------------------------
                # Move to next month
                # ---------------------------------------------

                if current_month == 12:

                    current_year += 1
                    current_month = 1

                else:

                    current_month += 1

            # -------------------------------------------------
            # End of run
            # -------------------------------------------------

            print(
                "\nMonthly Sentinel-2 composite "
                "workflow complete!"
            )

            print(
                "Run finished for ", GEOJSON_PATH, ":",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )

            print(
                "========================================"
            )

        finally:

            # Restore normal terminal output
            sys.stdout = original_stdout