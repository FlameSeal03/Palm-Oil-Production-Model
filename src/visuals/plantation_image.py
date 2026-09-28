from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.settings import IMAGE_DATE

from src.common.sentinel2_data import (
    blue,
    green,
    red,
    red_edge_1,
    nir,
    nir_2,
    swir_1,
)

from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.gndvi import calculate_gndvi
from src.indices.ndmi import calculate_ndmi
from src.indices.evi import calculate_evi
from src.indices.palm_mask import calculate_palm_mask


def normalize_band(
    band,
    lower_percentile=2,
    upper_percentile=98,
):
    """
    Normalize a satellite band for visualization.

    Percentile stretching improves visual contrast
    compared with displaying raw reflectance values.
    """

    valid_pixels = band[~np.isnan(band)]

    if valid_pixels.size == 0:
        raise ValueError(
            "No valid pixels were found in the satellite band."
        )

    lower = np.percentile(
        valid_pixels,
        lower_percentile,
    )

    upper = np.percentile(
        valid_pixels,
        upper_percentile,
    )

    if upper == lower:
        return np.zeros_like(band)

    normalized = (
        band - lower
    ) / (
        upper - lower
    )

    normalized = np.clip(
        normalized,
        0,
        1,
    )

    return normalized


def save_rgb_image(output_directory):
    """
    Create and save a true-color RGB image.

    Sentinel-2:
        Red   = B04
        Green = B03
        Blue  = B02
    """

    red_normalized = normalize_band(red)
    green_normalized = normalize_band(green)
    blue_normalized = normalize_band(blue)

    rgb_image = np.dstack(
        (
            red_normalized,
            green_normalized,
            blue_normalized,
        )
    )

    output_path = (
        output_directory
        / f"rgb_plantation_{IMAGE_DATE}.png"
    )

    plt.figure(figsize=(10, 10))

    plt.imshow(rgb_image)

    plt.title(
        f"Oil Palm Plantation - True Color\n"
        f"Sentinel-2: {IMAGE_DATE}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "RGB image saved:",
        output_path,
    )


def save_index_image(
    index,
    index_name,
    output_directory,
    vmin=-1,
    vmax=1,
):
    """
    Create and save a vegetation-index visualization.
    """

    output_path = (
        output_directory
        / f"{index_name.lower()}_plantation_{IMAGE_DATE}.png"
    )

    plt.figure(figsize=(10, 10))

    image = plt.imshow(
        index,
        vmin=vmin,
        vmax=vmax,
        cmap="RdYlGn",
    )

    plt.colorbar(
        image,
        label=index_name,
        fraction=0.046,
        pad=0.04,
    )

    plt.title(
        f"Oil Palm Plantation - {index_name}\n"
        f"Sentinel-2: {IMAGE_DATE}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"{index_name} image saved:",
        output_path,
    )

def save_palm_mask_image(
    palm_mask,
    output_directory,
):
    """
    Save the binary palm-tree candidate mask.

    White = likely palm/vegetation pixels
    Black = everything else
    """

    output_path = (
        output_directory
        / f"palm_mask_plantation_{IMAGE_DATE}.png"
    )

    plt.figure(figsize=(10, 10))

    plt.imshow(
        palm_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.title(
        f"Oil Palm Candidate Mask\n"
        f"Sentinel-2: {IMAGE_DATE}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Palm mask image saved:",
        output_path,
    )


if __name__ == "__main__":

    print(
        "Creating plantation visualizations..."
    )

    output_directory = Path("outputs")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # True-color RGB image
    # ----------------------------------------

    save_rgb_image(
        output_directory
    )

    # ----------------------------------------
    # Calculate existing vegetation indices
    # ----------------------------------------

    print("\nCalculating vegetation indices...")

    ndvi = calculate_ndvi(
        red,
        nir,
    )

    ndre = calculate_ndre(
        red_edge_1,
        nir_2,
    )

    gndvi = calculate_gndvi(
        green,
        nir,
    )

    ndmi = calculate_ndmi(
        nir,
        swir_1,
    )

    evi = calculate_evi(
        blue,
        red,
        nir,
    )

    palm_mask = calculate_palm_mask(
        ndvi,
        ndre,
        ndvi_threshold=0.70,
        ndre_threshold=0.50,
    )

    # ----------------------------------------
    # Save vegetation-index images
    # ----------------------------------------

    save_index_image(
        ndvi,
        "NDVI",
        output_directory,
    )

    save_index_image(
        ndre,
        "NDRE",
        output_directory,
    )

    save_index_image(
        gndvi,
        "GNDVI",
        output_directory,
    )

    save_index_image(
        ndmi,
        "NDMI",
        output_directory,
    )

    save_index_image(
        evi,
        "EVI",
        output_directory,
    )

    save_palm_mask_image(
        palm_mask,
        output_directory,
    )

    print(
        "\nPlantation visualizations complete!"
    )