import time

import numpy as np
import pystac_client
from shapely.geometry import shape

import sys
from pathlib import Path
from datetime import datetime

from config.settings import (
    GEOJSON_PATH,
    START_DATE,
    END_DATE,
)

from src.composites.process_sentinel2_scene import (
    process_sentinel2_scene,
)


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

CLOUD_THRESHOLD = 80.0

STAC_URL = (
    "https://stac.dataspace.copernicus.eu/v1"
)

COLLECTION = "sentinel-2-l2a"


# -------------------------------------------------------------
# Find scenes for one month
# -------------------------------------------------------------

def find_monthly_scenes(
    catalog,
    plantation_geometry,
    start_date,
    end_date,
):
    """
    Find Sentinel-2 scenes for a date range.

    Only scenes with cloud cover at or below
    CLOUD_THRESHOLD are retained.
    """

    search = catalog.search(
        collections=[COLLECTION],
        intersects=plantation_geometry,
        datetime=f"{start_date}/{end_date}",
    )

    scenes = list(search.items())

    print(
        f"\nScenes found between "
        f"{start_date} and {end_date}:",
        len(scenes),
    )

    # ---------------------------------------------------------
    # Filter by cloud cover
    # ---------------------------------------------------------

    usable_scenes = []

    for scene in scenes:

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

    # ---------------------------------------------------------
    # Arrays containing one layer per scene
    # ---------------------------------------------------------

    ndvi_arrays = []
    ndre_arrays = []
    gndvi_arrays = []
    ndmi_arrays = []
    evi_arrays = []

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
        # Process individual scene
        # -----------------------------------------------------

        results = process_sentinel2_scene(
            scene.id
        )

        ndvi_arrays.append(
            results["ndvi"]
        )

        ndre_arrays.append(
            results["ndre"]
        )

        gndvi_arrays.append(
            results["gndvi"]
        )

        ndmi_arrays.append(
            results["ndmi"]
        )

        evi_arrays.append(
            results["evi"]
        )

        # -----------------------------------------------------
        # Small delay to reduce API pressure
        # -----------------------------------------------------

        if index < len(scenes):
            time.sleep(3)

    # ---------------------------------------------------------
    # Stack scene arrays
    # ---------------------------------------------------------
    '''
    print(
        "\nStacking monthly scene arrays..."
    )
    '''
    ndvi_stack = np.stack(
        ndvi_arrays,
        axis=0,
    )

    ndre_stack = np.stack(
        ndre_arrays,
        axis=0,
    )

    gndvi_stack = np.stack(
        gndvi_arrays,
        axis=0,
    )

    ndmi_stack = np.stack(
        ndmi_arrays,
        axis=0,
    )

    evi_stack = np.stack(
        evi_arrays,
        axis=0,
    )

    # ---------------------------------------------------------
    # Calculate number of valid observations
    # ---------------------------------------------------------

    observation_count = np.sum(
        ~np.isnan(ndvi_stack),
        axis=0,
    )

    # ---------------------------------------------------------
    # Calculate pixel-wise monthly medians
    # ---------------------------------------------------------
    '''
    print(
        "Calculating monthly pixel-wise medians..."
    )
    '''
    monthly_ndvi = np.nanmedian(
        ndvi_stack,
        axis=0,
    )

    monthly_ndre = np.nanmedian(
        ndre_stack,
        axis=0,
    )

    monthly_gndvi = np.nanmedian(
        gndvi_stack,
        axis=0,
    )

    monthly_ndmi = np.nanmedian(
        ndmi_stack,
        axis=0,
    )

    monthly_evi = np.nanmedian(
        evi_stack,
        axis=0,
    )
    '''
    print(
        "Monthly composite created successfully!"
    )
    '''
    # ---------------------------------------------------------
    # Return composite
    # ---------------------------------------------------------

    return {
        "month": month_name,
        "ndvi": monthly_ndvi,
        "ndre": monthly_ndre,
        "gndvi": monthly_gndvi,
        "ndmi": monthly_ndmi,
        "evi": monthly_evi,
        "observation_count": observation_count,
    }


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

    print("\nNDVI:")
    print(
        "  Mean:",
        np.nanmean(composite["ndvi"]),
    )
    print(
        "  Median:",
        np.nanmedian(composite["ndvi"]),
    )
    print(
        "  Standard deviation:",
        np.nanstd(composite["ndvi"]),
    )

    print("\nNDRE:")
    print(
        "  Mean:",
        np.nanmean(composite["ndre"]),
    )
    print(
        "  Median:",
        np.nanmedian(composite["ndre"]),
    )
    print(
        "  Standard deviation:",
        np.nanstd(composite["ndre"]),
    )

    print("\nGNDVI:")
    print(
        "  Mean:",
        np.nanmean(composite["gndvi"]),
    )
    print(
        "  Median:",
        np.nanmedian(composite["gndvi"]),
    )
    print(
        "  Standard deviation:",
        np.nanstd(composite["gndvi"]),
    )

    print("\nNDMI:")
    print(
        "  Mean:",
        np.nanmean(composite["ndmi"]),
    )
    print(
        "  Median:",
        np.nanmedian(composite["ndmi"]),
    )
    print(
        "  Standard deviation:",
        np.nanstd(composite["ndmi"]),
    )

    print("\nEVI:")
    print(
        "  Mean:",
        np.nanmean(composite["evi"]),
    )
    print(
        "  Median:",
        np.nanmedian(composite["evi"]),
    )
    print(
        "  Standard deviation:",
        np.nanstd(composite["evi"]),
    )

    observation_count = composite[
        "observation_count"
    ]

    print(
        " Number of scenes:",
        np.nanmax(observation_count),
    )
    '''
    # ---------------------------------------------------------
    # Observation coverage
    # ---------------------------------------------------------

    observation_count = composite[
        "observation_count"
    ]

    print(
        "\nObservation count:"
    )

    print(
        "  Minimum:",
        np.nanmin(observation_count),
    )

    print(
        "  Maximum:",
        np.nanmax(observation_count),
    )

    print(
        "  Mean:",
        np.nanmean(observation_count),
    )

    '''

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
    # Open log file in append mode
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


            # -------------------------------------------------
            # Load STAC catalog
            # -------------------------------------------------

            catalog = pystac_client.Client.open(
                STAC_URL
            )

            print(
                "Connected to Copernicus Data Space."
            )

            # -------------------------------------------------
            # Load plantation geometry
            # -------------------------------------------------

            import json

            with open(
                GEOJSON_PATH,
                "r",
            ) as file:

                geojson = json.load(file)

            plantation_geometry = shape(
                geojson["features"][0]["geometry"]
            )

            print(
                "Plantation geometry loaded."
            )

            print(
                "Geometry type:",
                plantation_geometry.geom_type,
            )

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

                month_start = (
                    f"{current_year:04d}-"
                    f"{current_month:02d}-01"
                )

                if current_month == 12:

                    next_year = current_year + 1
                    next_month = 1

                else:

                    next_year = current_year
                    next_month = current_month + 1

                next_month_start = (
                    f"{next_year:04d}-"
                    f"{next_month:02d}-01"
                )

                # ---------------------------------------------
                # Find scenes for this month
                # ---------------------------------------------

                scenes = find_monthly_scenes(
                    catalog,
                    plantation_geometry,
                    month_start,
                    next_month_start,
                )

                # ---------------------------------------------
                # Create composite
                # ---------------------------------------------

                month_name = (
                    f"{current_year:04d}-"
                    f"{current_month:02d}"
                )

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

                current_year = next_year
                current_month = next_month

                # ---------------------------------------------
                # Delay between monthly searches
                # ---------------------------------------------

                time.sleep(3)

            # -------------------------------------------------
            # End of run
            # -------------------------------------------------

            print(
                "\nMonthly Sentinel-2 composite "
                "workflow complete!"
            )

            print(
                "Run finished:",
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