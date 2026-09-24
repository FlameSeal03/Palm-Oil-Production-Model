from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.settings import IMAGE_DATE
from src.common.sentinel2_data import (
    blue,
    green,
    red,
    nir,
)

from src.indices.ndvi import calculate_ndvi


def normalize_band(band, lower_percentile=2, upper_percentile=98):
    """
    Normalize a satellite band for visualization.

    Percentile stretching improves the visual contrast
    compared with displaying the raw reflectance values.
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
        / f"plantation_{IMAGE_DATE}_rgb.png"
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


def save_ndvi_image(output_directory):
    """
    Create and save an NDVI visualization.
    """

    ndvi = calculate_ndvi(
        red,
        nir,
    )

    output_path = (
        output_directory
        / f"plantation_{IMAGE_DATE}_ndvi.png"
    )

    plt.figure(figsize=(10, 10))

    image = plt.imshow(
        ndvi,
        vmin=-1,
        vmax=1,
        cmap="RdYlGn",
    )

    plt.colorbar(
        image,
        label="NDVI",
        fraction=0.046,
        pad=0.04,
    )

    plt.title(
        f"Oil Palm Plantation - NDVI\n"
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
        "NDVI image saved:",
        output_path,
    )


if __name__ == "__main__":

    print(
        "Creating plantation visualization..."
    )

    output_directory = Path("outputs")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_rgb_image(
        output_directory
    )

    save_ndvi_image(
        output_directory
    )

    print(
        "\nPlantation visualization complete!"
    )