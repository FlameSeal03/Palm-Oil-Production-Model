import numpy as np
import matplotlib.pyplot as plt

from src.common.sentinel2_data import (
    red,
    green,
    blue,
    red_edge_1,
    red_edge_2,
    red_edge_3,
    nir,
    nir_2,
    swir_1,
    swir_2,
)

from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.ndmi import calculate_ndmi
from src.indices.evi import calculate_evi
from src.indices.ndre_b06 import calculate_ndre_b06
from src.indices.ndre_b07 import calculate_ndre_b07


OUTPUT_PATH = "outputs/feature_comparison.png"


def normalize_index(index):
    """
    Normalize an index to the 0-1 range for visualization.

    The normalization is based on the minimum and maximum
    finite values in the current plantation image.
    """

    finite_values = index[np.isfinite(index)]

    minimum = np.min(finite_values)
    maximum = np.max(finite_values)

    if maximum == minimum:
        return np.zeros_like(index)

    normalized = (index - minimum) / (maximum - minimum)

    return normalized


def main():

    print("Calculating features for comparison visualization...")

    # Calculate the six features we currently want to compare.
    ndvi = calculate_ndvi(red, nir)

    ndre = calculate_ndre(
        red_edge_1,
        nir_2,
    )

    ndre_b06 = calculate_ndre_b06(
        red_edge_2,
        nir_2,
    )

    ndre_b07 = calculate_ndre_b07(
        red_edge_3,
        nir_2,
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

    features = [
        ("NDVI", ndvi),
        ("NDRE", ndre),
        ("NDRE-B06", ndre_b06),
        ("NDRE-B07", ndre_b07),
        ("NDMI", ndmi),
        ("EVI", evi),
    ]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes = axes.flatten()

    for axis, (name, index) in zip(axes, features):

        normalized = normalize_index(index)

        image = axis.imshow(
            normalized,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axis.set_title(name)
        axis.axis("off")

        figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

    figure.suptitle(
        "Sentinel-2 Feature Comparison — 2026-08-08",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"\nFeature comparison saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()