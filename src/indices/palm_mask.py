import numpy as np
import matplotlib.pyplot as plt

from src.common.sentinel2_data import (
    red,
    red_edge_1,
    red_edge_2,
    red_edge_3,
    nir,
    nir_2,
)

from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.ndre_b06 import calculate_ndre_b06
from src.indices.ndre_b07 import calculate_ndre_b07


OUTPUT_PATH = "outputs/palm_canopy_mask.png"


# Current thresholds
NDVI_THRESHOLD = 0.60
NDRE_THRESHOLD = 0.40
NDRE_B06_THRESHOLD = 0.10
NDRE_B07_THRESHOLD = 0.02


def calculate_palm_mask(
    ndvi,
    ndre,
    ndre_b06,
    ndre_b07,
    ndvi_threshold=NDVI_THRESHOLD,
    ndre_threshold=NDRE_THRESHOLD,
    ndre_b06_threshold=NDRE_B06_THRESHOLD,
    ndre_b07_threshold=NDRE_B07_THRESHOLD,
):
    """
    Create a rule-based oil-palm canopy candidate mask.

    White (1):
        Pixels meeting all four feature thresholds.

    Black (0):
        Pixels that do not meet the thresholds.

    This is a palm-canopy candidate mask, not a validated
    species classifier.
    """

    valid_pixels = (
        np.isfinite(ndvi)
        & np.isfinite(ndre)
        & np.isfinite(ndre_b06)
        & np.isfinite(ndre_b07)
    )

    ndvi_pass = (
        valid_pixels
        & (ndvi >= ndvi_threshold)
    )

    ndre_pass = (
        valid_pixels
        & (ndre >= ndre_threshold)
    )

    ndre_b06_pass = (
        valid_pixels
        & (ndre_b06 >= ndre_b06_threshold)
    )

    ndre_b07_pass = (
        valid_pixels
        & (ndre_b07 >= ndre_b07_threshold)
    )

    palm_mask = (
        ndvi_pass
        & ndre_pass
        & ndre_b06_pass
        & ndre_b07_pass
    )

    return (
        palm_mask.astype(np.uint8),
        valid_pixels,
        ndvi_pass,
        ndre_pass,
        ndre_b06_pass,
        ndre_b07_pass,
    )


def print_threshold_diagnostics(
    valid_pixels,
    ndvi,
    ndre,
    ndre_b06,
    ndre_b07,
    ndvi_pass,
    ndre_pass,
    ndre_b06_pass,
    ndre_b07_pass,
    palm_mask,
):
    """
    Print how many pixels pass each individual threshold
    and how many survive the combined conditions.
    """

    valid_count = np.count_nonzero(
        valid_pixels
    )

    print("\nThreshold diagnostics:")
    print("-" * 55)

    print(
        f"Valid pixels:        "
        f"{valid_count:>7,}"
    )

    print(
        f"NDVI >= {NDVI_THRESHOLD:.2f}:       "
        f"{np.count_nonzero(ndvi_pass):>7,} "
        f"({np.count_nonzero(ndvi_pass) / valid_count * 100:6.2f}%)"
    )

    print(
        f"NDRE >= {NDRE_THRESHOLD:.2f}:       "
        f"{np.count_nonzero(ndre_pass):>7,} "
        f"({np.count_nonzero(ndre_pass) / valid_count * 100:6.2f}%)"
    )

    print(
        f"NDRE-B06 >= {NDRE_B06_THRESHOLD:.2f}: "
        f"{np.count_nonzero(ndre_b06_pass):>7,} "
        f"({np.count_nonzero(ndre_b06_pass) / valid_count * 100:6.2f}%)"
    )

    print(
        f"NDRE-B07 >= {NDRE_B07_THRESHOLD:.2f}: "
        f"{np.count_nonzero(ndre_b07_pass):>7,} "
        f"({np.count_nonzero(ndre_b07_pass) / valid_count * 100:6.2f}%)"
    )

    print("-" * 55)

    # Add conditions one at a time.
    cumulative = valid_pixels.copy()

    conditions = [
        (
            "After NDVI",
            ndvi >= NDVI_THRESHOLD,
        ),
        (
            "After NDRE",
            ndre >= NDRE_THRESHOLD,
        ),
        (
            "After NDRE-B06",
            ndre_b06 >= NDRE_B06_THRESHOLD,
        ),
        (
            "After NDRE-B07",
            ndre_b07 >= NDRE_B07_THRESHOLD,
        ),
    ]

    print("\nCumulative filtering:")
    print("-" * 55)

    for name, condition in conditions:

        cumulative = (
            cumulative
            & condition
        )

        count = np.count_nonzero(
            cumulative
        )

        percentage = (
            count / valid_count * 100
        )

        print(
            f"{name:20s}: "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )

    print("-" * 55)

    final_count = np.count_nonzero(
        palm_mask
    )

    final_percentage = (
        final_count / valid_count * 100
    )

    print(
        f"\nFinal candidate mask: "
        f"{final_count:,} pixels "
        f"({final_percentage:.2f}%)"
    )

def save_mask(mask):
    """
    Save the binary palm-canopy mask as a black-and-white image.
    """

    figure, axis = plt.subplots(
        figsize=(10, 10)
    )

    axis.imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axis.set_title(
        "Oil Palm Canopy Candidate Mask — 2026-08-08"
    )

    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def print_feature_distributions(
    valid_pixels,
    ndvi,
    ndre,
    ndre_b06,
    ndre_b07,
):
    """
    Print percentile distributions for the features
    used by the palm-canopy candidate mask.
    """

    print("\nFeature distributions:")
    print("=" * 70)

    features = [
        ("NDVI", ndvi),
        ("NDRE", ndre),
        ("NDRE-B06", ndre_b06),
        ("NDRE-B07", ndre_b07),
    ]

    percentiles = [
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    for name, values in features:

        values = values[valid_pixels]

        percentile_values = np.percentile(
            values,
            percentiles,
        )

        print(f"\n{name}")
        print("-" * 40)

        for percentile, value in zip(
            percentiles,
            percentile_values,
        ):
            print(
                f"{percentile:>3}th percentile: "
                f"{value:.4f}"
            )


def main():

    print(
        "Calculating palm-canopy candidate mask..."
    )

    # Calculate the four features.
    ndvi = calculate_ndvi(
        red,
        nir,
    )

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

    (
        palm_mask,
        valid_pixels,
        ndvi_pass,
        ndre_pass,
        ndre_b06_pass,
        ndre_b07_pass,
    ) = calculate_palm_mask(
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )

    print_feature_distributions(
        valid_pixels,
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )

    print_threshold_diagnostics(
        valid_pixels,
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
        ndvi_pass,
        ndre_pass,
        ndre_b06_pass,
        ndre_b07_pass,
        palm_mask,
    )

    save_mask(
        palm_mask
    )

    print(
        f"\nPalm canopy mask saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()